/* ailaunch — run a program on the SpaceMiT K3 A100 ("AI") cores.
 *
 * Mechanism (per github.com/brucehoult/k3_ai): write your own decimal PID to
 * /proc/set_ai_thread, then execve the target. The write must happen BEFORE the
 * target starts: X100 (VLEN=256) and A100 (VLEN=1024) have different vector
 * lengths, so migrating a process that has already executed vector code is
 * unsafe — any VLEN-dependent state it computed is stale afterwards. execve
 * resets vector state, so the target starts clean with the A100's VLEN.
 *
 * Only raw syscalls are used after the write, for the same reason: no libc
 * routine (memcpy or printf dispatch) may run in the window between migration
 * and execve, because libc initialised while still on an X100 core.
 */
#include <unistd.h>
#include <sys/syscall.h>
#include <fcntl.h>

static void emit(const char *s, long n) { syscall(SYS_write, 2, s, n); }

int main(int argc, char **argv, char **envp) {
    if (argc < 2) { emit("usage: ailaunch <program> [args...]\n", 36); return 2; }

    long pid = syscall(SYS_getpid);
    char digits[24], out[26];
    int t = 0, n = 0;
    if (pid == 0) digits[t++] = '0';
    while (pid > 0) { digits[t++] = (char)('0' + (pid % 10)); pid /= 10; }
    while (t > 0) out[n++] = digits[--t];
    out[n++] = '\n';

    long fd = syscall(SYS_openat, AT_FDCWD, "/proc/set_ai_thread", O_WRONLY, 0);
    if (fd < 0) { emit("ailaunch: cannot open /proc/set_ai_thread\n", 42); return 1; }
    if (syscall(SYS_write, fd, out, n) != n) {
        emit("ailaunch: write to /proc/set_ai_thread failed\n", 46); return 1; }
    syscall(SYS_close, fd);

    syscall(SYS_execve, argv[1], argv + 1, envp);
    emit("ailaunch: execve failed\n", 24);
    return 127;
}
