
import os
from sys import stdout
from contextlib import contextmanager
@contextmanager
def stdout_redirected(to=os.devnull, STDOUT=None):
	"""
	Used in order to suppress annoying lsoda warning messages from scipy.integrate
	http://stackoverflow.com/a/22434262/190597 (J.F. Sebastian)
	"""

	def fileno(file_or_fd):
		fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
		if not isinstance(fd, int):
			raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
		return fd

	if STDOUT is None:
	   STDOUT = stdout

	stdout_fd = fileno(STDOUT)
	# copy stdout_fd before it is overwritten
	#NOTE: `copied` is inheritable on Windows when duplicating a standard stream
	with os.fdopen(os.dup(stdout_fd), 'wb') as copied: 
		STDOUT.flush()  # flush library buffers that dup2 knows nothing about
		try:
			os.dup2(fileno(to), stdout_fd)  # $ exec >&to
		except ValueError:  # filename
			with open(to, 'wb') as to_file:
				os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
		try:
			yield STDOUT # allow code to be run with the redirected stdout
		finally:
			# restore stdout to its previous value
			#NOTE: dup2 makes stdout_fd inheritable unconditionally
			STDOUT.flush()
			os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied