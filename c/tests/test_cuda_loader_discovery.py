"""Exercise discovery before initialization with native stubs, without GPU SDKs.

Module-state probes and the HIP refusal control are adapted from ZhiyangK's
PR #1579 (https://github.com/JustVugg/colibri/pull/1579). Unlike its old-DLL
fallback test, these stubs model device_count as initialized contexts and
require a diagnostic when the discovery export is absent (#1542).
"""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


HERE = Path(__file__).resolve().parent.parent


@unittest.skipUnless(os.name == "nt" and shutil.which("gcc"), "requires Windows and MinGW gcc")
class CudaLoaderDiscoveryTest(unittest.TestCase):
    def run_probe(self, discovery=True, backend=True, hip=False, visible=2):
        with tempfile.TemporaryDirectory(prefix="coli discovery ") as tmp:
            root = Path(tmp)
            host = root / "probe.c"
            host.write_text('''#include <stdio.h>
#include <windows.h>
#include "backend_cuda.h"
#ifdef COLI_HIP_DLL
#define BACKEND_DLL L"coli_hip.dll"
#else
#define BACKEND_DLL L"coli_cuda.dll"
#endif
int main(void) {
    int device = 0;
    printf("loaded_before=%d\\n", GetModuleHandleW(BACKEND_DLL) != NULL);
    printf("visible=%d\\n", coli_cuda_available_device_count());
    printf("loaded_after=%d\\n", GetModuleHandleW(BACKEND_DLL) != NULL);
    printf("before=%d\\n", coli_cuda_device_count());
    printf("init=%d\\n", coli_cuda_init(&device, 1));
    printf("after=%d\\n", coli_cuda_device_count());
    printf("visible_after_init=%d\\n", coli_cuda_available_device_count());
    coli_cuda_shutdown();
    return 0;
}
''', encoding="ascii")
            def build(*args):
                proc = subprocess.run(["gcc", *map(str, args)], capture_output=True,
                                      text=True, timeout=60)
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            defines = ["-DCOLI_CUDA"] + (["-DCOLI_HIP_DLL"] if hip else [])
            build(*defines, "-I", HERE, host, HERE / "backend_loader.c",
                  "-o", root / "probe.exe")
            if backend:
                source = (HERE / "backend_loader.c").read_text(encoding="utf-8")
                names = re.findall(r"^\s+RESOLVE(?:_OPT)?\((\w+),", source, re.M)
                bodies = {
                    "init": "int coli_cuda_init(const int *d, int n) { (void)d; count=n; return 1; }",
                    "shutdown": "void coli_cuda_shutdown(void) { count=0; }",
                    "device_count": "int coli_cuda_device_count(void) { return count; }",
                    "available_device_count": "int coli_cuda_available_device_count(void) { return %d; }" % visible,
                }
                lines = ["static int count;"]
                for name in names:
                    if name == "available_device_count" and not discovery:
                        continue
                    body = bodies.get(name, "int coli_cuda_%s(void) { return 0; }" % name)
                    lines.append("__declspec(dllexport) " + body)
                dll = root / "stub.c"
                dll.write_text("\n".join(lines), encoding="ascii")
                build("-shared", dll, "-o", root / ("coli_hip.dll" if hip else "coli_cuda.dll"))
            env = os.environ.copy()
            for key in ("COLI_GPUS", "COLI_GPU", "COLI_HIP_RUNTIME_DIR"):
                env.pop(key, None)
            proc = subprocess.run([str(root / "probe.exe")], capture_output=True,
                                  text=True, timeout=30, cwd=root, env=env)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            return proc, dict(line.split("=", 1) for line in proc.stdout.splitlines())

    def assert_probe(self, proc, out, *, loaded, visible, initialized):
        self.assertEqual(out, {
            "loaded_before": "0", "visible": str(visible),
            "loaded_after": str(loaded), "before": "0",
            "init": str(initialized), "after": str(initialized),
            "visible_after_init": str(visible),
        }, "\nstdout:\n%s\nstderr:\n%s" % (proc.stdout, proc.stderr))

    def test_discovery_precedes_init(self):
        proc, out = self.run_probe()
        self.assert_probe(proc, out, loaded=1, visible=2, initialized=1)
        self.assertNotIn("missing symbol", proc.stderr)

    def test_old_dll_explains_missing_discovery_but_allows_explicit_init(self):
        proc, out = self.run_probe(discovery=False)
        # Explicit init creates one context, but discovery must still report
        # zero afterwards: that context is not a substitute for this export.
        self.assert_probe(proc, out, loaded=1, visible=0, initialized=1)
        self.assertIn("missing symbol coli_cuda_available_device_count", proc.stderr)
        self.assertIn("rebuild the backend DLL", proc.stderr)
        self.assertIn("COLI_GPUS", proc.stderr)

    def test_missing_dll_returns_zero(self):
        proc, out = self.run_probe(backend=False)
        self.assert_probe(proc, out, loaded=0, visible=0, initialized=0)
        self.assertIn("could not be loaded", proc.stderr)

    def test_zero_visible_devices_is_not_a_missing_export(self):
        # A valid zero from the export must not be confused with load failure
        # or an old DLL. The fake init deliberately succeeds independently.
        proc, out = self.run_probe(visible=0)
        self.assert_probe(proc, out, loaded=1, visible=0, initialized=1)
        self.assertNotIn("missing symbol", proc.stderr)
        self.assertNotIn("could not be loaded", proc.stderr)

    def test_hip_discovery_keeps_runtime_configuration_guard(self):
        # Even with a backend present, probing must refuse before mapping it
        # when the required HIP runtime directory is unset.
        proc, out = self.run_probe(hip=True)
        self.assert_probe(proc, out, loaded=0, visible=0, initialized=0)
        self.assertIn("COLI_HIP_RUNTIME_DIR is not set", proc.stderr)
        self.assertNotIn("missing symbol", proc.stderr)


if __name__ == "__main__":
    unittest.main()
