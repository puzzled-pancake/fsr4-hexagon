/* live_win.c — WinSock transport for the proxy live mode.
 * Separate TU so <winsock2.h> is included cleanly (windows.h in the
 * concatenated proxy TU pulls winsock1; mixing the two headers in one TU
 * conflicts). Exposes minimal blocking send/recv with timeouts. */
#include <winsock2.h>
#include <ws2tcpip.h>
#include <errno.h>

static int g_wsa_done = 0;

int live_net_startup(void) {
 if (g_wsa_done) return 0;
 WSADATA w;
 if (WSAStartup(MAKEWORD(2, 2), &w) != 0) return -1;
 g_wsa_done = 1;
 return 0;
}

/* connect with a hard timeout (non-blocking connect + select) */
int live_net_connect(const char* host, int port, int timeout_ms) {
 if (live_net_startup() != 0) return -1;
 SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
 if (s == INVALID_SOCKET) return -1;
 unsigned long nb = 1;
 ioctlsocket(s, FIONBIO, &nb);
 struct sockaddr_in sa;
 memset(&sa, 0, sizeof sa);
 sa.sin_family = AF_INET;
 sa.sin_port = htons((unsigned short)port);
 inet_pton(AF_INET, host, &sa.sin_addr);
 int r = connect(s, (struct sockaddr*)&sa, sizeof sa);
 if (r != 0) {
 if (WSAGetLastError() != WSAEWOULDBLOCK) { closesocket(s); return -1; }
 fd_set wset; FD_ZERO(&wset); FD_SET(s, &wset);
 struct timeval tv = { timeout_ms / 1000, (timeout_ms % 1000) * 1000 };
 if (select(0, NULL, &wset, NULL, &tv) != 1) { closesocket(s); return -1; }
 }
 nb = 0; ioctlsocket(s, FIONBIO, &nb); /* back to blocking */
 /* >= the daemon's 8.3MB response frame — a 4MB window made the
 * daemon's mid-frame send stall waiting for the proxy to drain */
 int buf = 16 * 1024 * 1024;
 setsockopt(s, SOL_SOCKET, SO_RCVBUF, (const char*)&buf, sizeof buf);
 setsockopt(s, SOL_SOCKET, SO_SNDBUF, (const char*)&buf, sizeof buf);
 int nd = 1; setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (const char*)&nd, sizeof nd);
 /* 15 s: live in-game frames measured 20-150 ms steady but game stalls
 * (loading, ENOSPC) hit 5.7-8.8 s — the old 3 s killed live mid-session */
 int to = 15000;
 setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char*)&to, sizeof to);
 setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, (const char*)&to, sizeof to);
 return (int)s;
}

int live_net_send_all(int s, const char* b, size_t n) {
 size_t put = 0;
 while (put < n) {
 int r = send((SOCKET)s, b + put, (int)(n - put), 0);
 if (r == SOCKET_ERROR) { if (WSAGetLastError() == WSAEINTR) continue; return -1; }
 put += (size_t)r;
 }
 return 0;
}

int live_net_recv_all(int s, char* b, size_t n) {
 size_t got = 0;
 while (got < n) {
 int r = recv((SOCKET)s, b + got, (int)(n - got), 0);
 if (r == SOCKET_ERROR) { if (WSAGetLastError() == WSAEINTR) continue; return -1; }
 if (r == 0) return -1;
 got += (size_t)r;
 }
 return 0;
}

void live_net_close(int s) {
 if (s >= 0) closesocket((SOCKET)s);
}

/* abort a socket so OTHER threads blocked on it wake. On Wine,
 * closesocket() alone does not abort another thread's pending recv/send —
 * shutdown(SD_BOTH) does, on both native Winsock and Wine. */
void live_net_abort(int s) {
 if (s >= 0) {
 shutdown((SOCKET)s, SD_BOTH);
 closesocket((SOCKET)s);
 }
}
