import os, time
import multiprocessing
import subprocess
from dataclasses import dataclass, field
from typing import List

@dataclass
class ANTSStage:
    transform: str
    transform_args: str
    metrics: str
    metrics_args: str
    convergence: str
    shrink_factors: str
    smoothing_sigmas: str

@dataclass
class ANTSParams:
    use_float: bool = False
    interpolation: str = "Linear"
    histogram_matching: bool = False
    winsorize: List = field(default_factory=lambda: [0.0, 1.0])
    init_transform_fx_to_mv: str = None
    stages: List[ANTSStage] = None

class ANTSRegisteration:
    def __init__(self, params: ANTSParams, verbose=True, ants_path=None, num_threads=None):
        # default settings
        self.params = params
        self.verbose = verbose
        self.use_mask = False
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

    def set_fixed_image(self, path, voxel_size):
        """
        Define path to the fixed image
        """
        if not os.path.exists(path):
            msg = path + " does not exist!"
            raise FileNotFoundError(msg)
        self.fixed_image_path = path
        self.fixed_image_vxsize = float(voxel_size)

    def set_moving_image(self, path, voxel_size):
        """
        Define path to the moving image
        """
        if not os.path.exists(path):
            msg = path + " does not exist!"
            raise FileNotFoundError(msg)
        self.moving_image_path = path
        self.moving_image_vxsize = float(voxel_size)
    
    def set_outputs(self, outdir):
        """
        Define a directory to store outputs
        INPUTS:
        outdir
        """
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        self.outdir = outdir
        # Define output filenames
        self.warpfield = str(os.path.join(self.outdir, "F2M_"))
        self.warped = str(os.path.join(self.outdir, "Warped.nii.gz"))
        self.invwarped = str(os.path.join(self.outdir, "InvWarped.nii.gz"))
    
    def set_mask(self, fixed_mask_path, moving_mask_path):
        """
        Define masks of the fixed and moving image.
        """
        if not os.path.exists(fixed_mask_path):
            raise ValueError(f"{fixed_mask_path} does not exist!")
        if not os.path.exists(moving_mask_path):
            raise ValueError(f"{moving_mask_path} does not exist!")
        self.fixed_mask_path = fixed_mask_path
        self.moving_mask_path = moving_mask_path
        self.use_mask = True

    def run_shell_command(self, cmd, env=None, verbose=True):
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
    
    def compile_ants_options(self):
        opt = []
        opt.extend(["--dimensionality", "3"])

        if self.params.init_transform_fx_to_mv is not None:
            init = ["--initial-moving-transform",  f"{self.params.init_transform_fx_to_mv}"]
        else:
            init = ["--initial-moving-transform",  f"[{self.fixed_image_path},{self.moving_image_path},0]"]
        opt.extend(init)

        opt.extend(["--interpolation", self.params.interpolation])
        opt.extend(["--use-histogram-matching", str(int(self.params.histogram_matching))])
        #opt.extend(["--winsorize-image-intensities", ])
        opt.extend(["--float", str(int(self.params.use_float))])
        opt.extend(["--verbose", str(int(self.verbose))])
        opt.extend(["--output", f"[{self.warpfield},{self.warped},{self.invwarped}]"])

        return opt

    def compile_registration_stages(self):
        """
        define ANTS registration stages
        """
        cmd = []
        for st in self.params.stages:
            if st.transform in ["Translation", "Rigid", "Affine", "SyN"]:
                cmd.extend(["--transform", f"{st.transform}[{st.transform_args}]"])
            else:
                raise ValueError("Unrecognized stage type!:" + st.transform)
            cmd.extend(["--metric", f"{st.metrics}[{self.fixed_image_path},{self.moving_image_path},{st.metrics_args}]"])
            cmd.extend(["--convergence", f"[{st.convergence}]"])
            cmd.extend(["--shrink-factors", f"{st.shrink_factors}"])
            cmd.extend(["--smoothing-sigmas",  f"{st.smoothing_sigmas}"])
            if self.use_mask:
                cmd.extend(["-x", f"[{self.fixed_mask_path},{self.moving_mask_path}]"])

        return cmd
    
    def run(self):
        myenv = os.environ.copy()
        myenv["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(self.num_threads)

        opt = self.compile_ants_options()
        stages = self.compile_registration_stages()
        cmd = [str(os.path.join(self.ants_path, "antsRegistration"))] + opt + stages

        if self.verbose:
            print("#"*40)
            print(cmd)
            print("#"*40 + "\n")

        start_time = time.time()
        rc = self.run_shell_command(cmd, env=myenv, verbose=True)
        elapsed_time = time.time() - start_time

        if rc==0:
            print("\n\nANTS registration successfully completed!")
            print(f"Total elapsed time: {elapsed_time:.2f} s")
        else:
            print("\n\nANTS registration failed!")
            print("Return code: ", rc)
