import multiprocessing
import os
from dataclasses import dataclass, field
from .helper import run_shell_command

@dataclass
class ANTSTransformationStages:
    transformation: str
    inverse: bool

@dataclass
class ANTSTransformationParams:
    interpolation: str = "Linear"

class ANTSImageTransformation:

    def __init__(self, params:ANTSTransformationParams = None, verbose=True, ants_path=None, num_threads=None):
        if params is None:
            self.params = ANTSTransformationParams()
        else:
            self.params = params
        self.verbose = verbose
        if ants_path is None:
            self.ants_path = ''
        else:
            # check if this path is OK
            antsexec = os.path.join(ants_path, 'antsRegistration')
            if not os.path.exists(antsexec):
                msg = 'ANTs path error! Executable files were not found in' + antsexec
                raise ValueError(msg)
            self.ants_path = ants_path
        if num_threads is None:
            self.num_threads = multiprocessing.cpu_count()
        else:
            self.num_threads = num_threads
    
    def set_reference_image(self, path):
        """
        Define path to the reference image
        """
        if not os.path.exists(path):
            msg = path + " does not exist!"
            raise FileNotFoundError(msg)
        self.reference_image_path = path

    def set_moving_image(self, path):
        """
        Define path to the moving image
        """
        if not os.path.exists(path):
            msg = path + " does not exist!"
            raise FileNotFoundError(msg)
        self.moving_image_path = path
    
    def set_outname(self, outname):
        """
        Define path of the output
        """
        outdir = os.path.dirname(outname)
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        self.outname = outname

    def set_transformations(self, stages):
        transformations = []
        for st in stages:
            if st.inverse:
                t = ["-t",  f"[{st.transformation},1]"]
            else:
                t = ["-t", f"{st.transformation}"]
            transformations += t

        self.transformations = transformations

    def run(self):
        myenv = os.environ.copy()
        myenv["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(self.num_threads)
    
        cmd = [str(os.path.join(self.ants_path, "antsApplyTransforms"))]
        cmd.extend(["-i", self.moving_image_path])
        cmd.extend(["-r", self.reference_image_path])
        cmd.extend(["-o", self.outname])
        cmd.extend(["--input-image-type", "0"])
        cmd.extend(["--float", "1"])
        cmd.extend(["--interpolation", f"{self.params.interpolation}"])
        cmd.extend(self.transformations)

        print(cmd)
        run_shell_command(cmd, env=myenv, verbose=True)

class ANTSPointTransformation:

    def __init__(self, verbose=True, ants_path=None, num_threads=None):
        self.verbose = verbose
        if ants_path is None:
            self.ants_path = ''
        else:
            # check if this path is OK
            antsexec = os.path.join(ants_path, 'antsRegistration')
            if not os.path.exists(antsexec):
                msg = 'ANTs path error! Executable files were not found in' + antsexec
                raise ValueError(msg)
            self.ants_path = ants_path
        if num_threads is None:
            self.num_threads = multiprocessing.cpu_count()
        else:
            self.num_threads = num_threads

    def set_moving_points(self, path):
        """
        Define path to the moving points, given as CSV file
        """
        if not os.path.exists(path):
            msg = path + " does not exist!"
            raise FileNotFoundError(msg)
        self.moving_points_path = path

    def set_outname(self, outname):
        """
        Define the path of the output
        """
        outdir = os.path.dirname(outname)
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        self.outname = outname

    def set_transformations(self, stages):
        transformations = []
        for st in stages:
            if st.inverse:
                t = ["-t",  f"[{st.transformation},1]"]
            else:
                t = ["-t", f"{st.transformation}"]
            transformations += t

        self.transformations = transformations

    def run(self):
        myenv = os.environ.copy()
        myenv["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(self.num_threads)

        cmd = [str(os.path.join(self.ants_path, "antsApplyTransformsToPoints"))]
        cmd.extend(["-d", "3"])
        cmd.extend(["-i", self.moving_points_path])
        cmd.extend(["-o", self.outname])
        cmd.extend(self.transformations)

        print(cmd)
        run_shell_command(cmd, env=myenv, verbose=True)
