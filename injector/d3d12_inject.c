/* d3d12_inject.c — d3d12.dll seat that applies the NGX EnableSignatureOverride
 * registry values (from the NVIDIA DLSS SDK regs/EnableSignatureOverride.reg)
 * at process load, BEFORE the game's DLSS snippet probe. In DX11 mode the game
 * only probes d3d12 availability — all exports return failure, which matches
 * the previous (builtin vkd3d failure) behavior exactly.
 *
 * Registry values written (per the SDK's EnableSignatureOverride.reg):
 *   HKLM\SOFTWARE\NVIDIA Corporation\Global            {41FCC608-8496-4DEF-B43E-7D9BD675A6FF} = REG_BINARY 01
 *   HKLM\SYSTEM\ControlSet001\Services\nvlddmkm        {41FCC608-8496-4DEF-B43E-7D9BD675A6FF} = REG_BINARY 01
 *
 * Build: x86_64-w64-mingw32-gcc -O2 -shared -o d3d12.dll d3d12_inject.c d3d12_inject.def -static-libgcc
 */
#include <windows.h>
#include <stdio.h>
#include <stdarg.h>
#include <ctype.h>
#include <string.h>

static FILE* g_log = NULL;
static CRITICAL_SECTION g_cs;
static LONG g_csinit = 0;

static void LOG(const char* fmt, ...) {
    if (!g_csinit) { InitializeCriticalSection(&g_cs); g_csinit = 1; }
    EnterCriticalSection(&g_cs);
    if (!g_log) {
        CreateDirectoryA("D:\\fsr4cap", NULL);
        g_log = fopen("D:\\fsr4cap\\d3d12_inject.log", "a");
        if (!g_log) { LeaveCriticalSection(&g_cs); return; }
        fprintf(g_log, "=== d3d12 injector attached ===\n");
    }
    {
        va_list ap; va_start(ap, fmt);
        SYSTEMTIME st; GetLocalTime(&st);
        fprintf(g_log, "[%02d:%02d:%02d.%03d] ", st.wHour, st.wMinute, st.wSecond, st.wMilliseconds);
        vfprintf(g_log, fmt, ap);
        fprintf(g_log, "\n");
        fflush(g_log);
        va_end(ap);
    }
    LeaveCriticalSection(&g_cs);
}

static const char* SIG_GUID = "{41FCC608-8496-4DEF-B43E-7D9BD675A6FF}";

static void write_override_flag(HKEY root, const char* path) {
    HKEY k = NULL;
    LONG r = RegCreateKeyExA(root, path, 0, NULL, REG_OPTION_NON_VOLATILE, KEY_SET_VALUE, NULL, &k, NULL);
    if (r != ERROR_SUCCESS || !k) {
        LOG("reg create FAIL %s err=%ld", path, (long)r);
        return;
    }
    const BYTE one = 0x01;
    r = RegSetValueExA(k, SIG_GUID, 0, REG_BINARY, &one, 1);
    LOG("reg set %s\\...%s -> %ld", path, SIG_GUID, (long)r);
    RegCloseKey(k);
}

static void apply_signature_override(void) {
    LOG("applying EnableSignatureOverride");
    write_override_flag(HKEY_LOCAL_MACHINE, "SOFTWARE\\NVIDIA Corporation\\Global");
    write_override_flag(HKEY_LOCAL_MACHINE, "SYSTEM\\ControlSet001\\Services\\nvlddmkm");
    /* NGX\Core variant as well — some loader versions read the flag here */
    write_override_flag(HKEY_LOCAL_MACHINE, "SOFTWARE\\NVIDIA Corporation\\Global\\NGX\\Core");
}

/* ---- graphics-settings discovery: dump any game publisher key under
 * HKCU\Software so the exact value names for a later AO/ScreenEffects
 * patch are known (research: settings live in the registry, not .ini) ---- */

static int name_has(const char* name, const char* pat) {
    size_t nl = strlen(name), pl = strlen(pat);
    if (pl == 0 || pl > nl) return 0;
    for (size_t a = 0; a + pl <= nl; a++) {
        size_t b = 0;
        while (b < pl && tolower((unsigned char)name[a + b]) == tolower((unsigned char)pat[b])) b++;
        if (b == pl) return 1;
    }
    return 0;
}

static void dump_reg_values(HKEY k, const char* path, FILE* f) {
    DWORD i = 0;
    for (;;) {
        char name[256]; DWORD nlen = sizeof(name);
        BYTE data[1024]; DWORD dlen = sizeof(data); DWORD type = 0;
        LONG r = RegEnumValueA(k, i++, name, &nlen, NULL, &type, data, &dlen);
        if (r != ERROR_SUCCESS) break;
        if (type == REG_DWORD && dlen == 4) {
            fprintf(f, "%s\\%s = DWORD %u (0x%08x)\n", path, name,
                    *(unsigned*)data, *(unsigned*)data);
        } else if (type == REG_SZ && dlen > 0 && dlen <= sizeof(data)) {
            data[dlen - 1] = 0;
            fprintf(f, "%s\\%s = SZ \"%s\"\n", path, name, (char*)data);
        } else {
            fprintf(f, "%s\\%s = type %lu, %lu bytes\n", path, name, (long)type, (long)dlen);
        }
    }
}

static void dump_reg_tree(HKEY root, const char* path, int depth, FILE* f) {
    HKEY k = NULL;
    if (RegOpenKeyExA(root, path, 0, KEY_READ, &k) != ERROR_SUCCESS || !k) {
        fprintf(f, "%s <open failed>\n", path);
        return;
    }
    dump_reg_values(k, path, f);
    if (depth > 0) {
        DWORD i = 0;
        for (;;) {
            char name[256]; DWORD nlen = sizeof(name);
            LONG r = RegEnumKeyExA(k, i++, name, &nlen, NULL, NULL, NULL, NULL);
            if (r != ERROR_SUCCESS) break;
            char sub[512];
            snprintf(sub, sizeof(sub), "%s\\%s", path, name);
            dump_reg_tree(root, sub, depth - 1, f);
        }
    }
    RegCloseKey(k);
}

static void dump_graphics_registry(void) {
    FILE* f = fopen("D:\\fsr4cap\\registry_dump.txt", "w");
    if (!f) { LOG("registry dump: open D:\\fsr4cap\\registry_dump.txt FAILED"); return; }
    HKEY sw = NULL;
    if (RegOpenKeyExA(HKEY_CURRENT_USER, "Software", 0, KEY_READ, &sw) == ERROR_SUCCESS && sw) {
        DWORD i = 0;
        for (;;) {
            char name[256]; DWORD nlen = sizeof(name);
            LONG r = RegEnumKeyExA(sw, i++, name, &nlen, NULL, NULL, NULL, NULL);
            if (r != ERROR_SUCCESS) break;
            if (name_has(name, "crystal") || name_has(name, "tomb") || name_has(name, "square") ||
                name_has(name, "eidos") || name_has(name, "nixxes") || name_has(name, "feral")) {
                char sub[300];
                snprintf(sub, sizeof(sub), "Software\\%s", name);
                fprintf(f, "=== %s ===\n", sub);
                dump_reg_tree(HKEY_CURRENT_USER, sub, 2, f);
            }
        }
        RegCloseKey(sw);
    } else {
        fprintf(f, "HKCU\\Software <open failed>\n");
    }
    fclose(f);
    LOG("registry dump written (publisher keys under HKCU\\Software)");
}

/* ---- RotTR graphics hygiene (community Xmas-artifact fixes, research-confirmed).
 * Value names verified live via registry_dump.txt (d3d12 ATTACH precedes the
 * game's settings read: probe -> decision -> settings, same-session effect).
 * The game may re-save on exit, so we re-apply every launch. ---- */

static void patch_rottr_graphics(void) {
    static const struct { const char* name; DWORD v; } vals[] = {
        { "ScreenEffects", 0 },            /* the documented Xmas-artifact fix   */
        { "FilmGrain", 0 },                /* pre-upscale noise injection        */
        { "LensFlares", 0 },               /* pre-upscale flare pass             */
        { "AmbientOcclusionQuality", 0 },  /* pin AO Off (detaches GFSDK paths)  */
        { "AsyncCompute", 0 },             /* Inf-flood suspect #1: DLSS-path async
                                            * compute fencing on Turnip TBDR — the
                                            * composite may read zero-filled inputs
                                            * and divide -> blanket +/-Inf         */
        { "HighPrecisionRT", 0 },          /* Inf-flood suspect #2: alt RT path   */
        /* fresh-container enable — the in-menu toggle reverts before NGX
         * CreateFeature (game-side capability screen); the old container only
         * worked because DLSS=1 persisted in the registry from the first-ever
         * enable. Force it at ATTACH so the game boots with DLSS already on. */
        { "DLSS", 1 },
        { "DLSS Previous", 1 },
        { "FXAA", 0 },                     /* don't fight the DLSS flag          */
    };
    HKEY k = NULL;
    LONG r = RegCreateKeyExA(HKEY_CURRENT_USER,
        "Software\\Crystal Dynamics\\Rise of the Tomb Raider\\Graphics",
        0, NULL, REG_OPTION_NON_VOLATILE, KEY_SET_VALUE, NULL, &k, NULL);
    if (r != ERROR_SUCCESS || !k) {
        LOG("rottr graphics key OPEN FAIL %ld", (long)r);
        return;
    }
    for (unsigned i = 0; i < sizeof(vals) / sizeof(vals[0]); i++) {
        DWORD v = vals[i].v;
        LONG wr = RegSetValueExA(k, vals[i].name, 0, REG_DWORD, (const BYTE*)&v, sizeof(v));
        LOG("rottr patch %s = %u -> %ld", vals[i].name, (unsigned)v, (long)wr);
    }
    RegCloseKey(k);
}

/* ---- d3d12 surface: probe-time failure, identical outcome to the current
 *      builtin vkd3d failure on this stack ---- */

HRESULT __declspec(dllexport) D3D12CreateDevice(void* pAdapter, unsigned int FeatureLevel, void* RIID, void** ppDevice) {
    LOG("D3D12CreateDevice called -> E_FAIL (DX11-mode stub)");
    (void)pAdapter; (void)FeatureLevel; (void)RIID; (void)ppDevice;
    return (HRESULT)0x887A0005L; /* DXGI_ERROR_UNSUPPORTED */
}

HRESULT __declspec(dllexport) D3D12GetDebugInterface(void* riid, void** ppvDebug) {
    (void)riid; (void)ppvDebug;
    return (HRESULT)0x80004001L; /* E_NOTIMPL */
}

HRESULT __declspec(dllexport) D3D12SerializeRootSignature(const void* pRootSignature, unsigned int Version, void** ppBlob, void** ppError) {
    (void)pRootSignature; (void)Version; (void)ppBlob; (void)ppError;
    return (HRESULT)0x80004001L;
}

HRESULT __declspec(dllexport) D3D12CreateRootSignatureDeserializer(void* pSrcData, unsigned int SrcDataSizeInBytes, void* pRootSignatureDeserializerInterface, void** ppRootSignatureDeserializer) {
    (void)pSrcData; (void)SrcDataSizeInBytes; (void)pRootSignatureDeserializerInterface; (void)ppRootSignatureDeserializer;
    return (HRESULT)0x80004001L;
}

HRESULT __declspec(dllexport) D3D12SerializeVersionedRootSignature(const void* pRootSignature, void** ppBlob, void** ppError) {
    (void)pRootSignature; (void)ppBlob; (void)ppError;
    return (HRESULT)0x80004001L;
}

HRESULT __declspec(dllexport) D3D12CreateVersionedRootSignatureDeserializer(void* pSrcData, unsigned int SrcDataSizeInBytes, void* pRootSignatureDeserializerInterface, void** ppRootSignatureDeserializer) {
    (void)pSrcData; (void)SrcDataSizeInBytes; (void)pRootSignatureDeserializerInterface; (void)ppRootSignatureDeserializer;
    return (HRESULT)0x80004001L;
}

HRESULT __declspec(dllexport) D3D12EnableExperimentalFeatures(unsigned int NumFeatures, const void* pIIDs, void* pConfigurationStructs, unsigned int* pConfigurationStructSizes) {
    (void)NumFeatures; (void)pIIDs; (void)pConfigurationStructs; (void)pConfigurationStructSizes;
    return (HRESULT)0x80004001L;
}

void* __declspec(dllexport) D3D12GetInterface(const void* rclsid, const void* riid) {
    (void)rclsid; (void)riid;
    return NULL;
}

BOOL WINAPI DllMain(HINSTANCE hinst, DWORD reason, LPVOID reserved) {
    (void)reserved;
    if (reason == DLL_PROCESS_ATTACH) {
        DisableThreadLibraryCalls(hinst);
        LOG("DllMain ATTACH");
        apply_signature_override();
        if (getenv("FSR4_DUMP_REG")) dump_graphics_registry();  /* research aid, off by default */
        patch_rottr_graphics();
    }
    return TRUE;
}
