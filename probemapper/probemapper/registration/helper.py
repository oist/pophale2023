import numpy as np
from scipy import io
import os, subprocess

def define_initial_transformation(aff, transl, outname):
	"""
	generates a .mat file defining the initial transformation
	INPUTS:
	aff: 3x3 numpy matrix
	transl: 3x1 vector
	outname: .mat file name 
	"""
	aff = aff.reshape((9,1))
	transl = transl.reshape((3,1))
	m = np.concatenate((aff, transl))
	mat = {'AffineTransform_double_3_3': m,
          'fixed': np.array([[0.],[0.],[0.]])}
	# .mat format must be v4! (compatible with before 7.2)
	# https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.savemat.html
	io.savemat(outname, mat, format='4')
	print(".mat file saved as", outname)

def run_shell_command(cmd, env=None, verbose=True):
	"""
	Runs a command on the shell. The output is printed 
	as soon as stdout buffer is flushed
	"""
	if env is None:
		env = os.environ.copy()
	pr = subprocess.Popen(cmd, env=env, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	while pr.poll() is None:
		line = pr.stdout.readline()
		if line != '':
			if verbose: print(line.decode('utf-8').rstrip())

	# Sometimes the process exits before we have all of the output, so
	# we need to gather the remainder of the output.
	remainder = pr.communicate()[0]
	if remainder:
		if verbose: print(remainder.decode('utf-8').rstrip())

	rc = pr.poll()
	return rc
