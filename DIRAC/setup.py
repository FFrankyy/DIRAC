from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    cmdclass = {'build_ext':build_ext},
    ext_modules=[
                 Extension('PrepareBatchGraph', sources=['PrepareBatchGraph.pyx', './src/lib/PrepareBatchGraph.cpp', './src/lib/graph.cpp', './src/lib/graph_struct.cpp'], language='c++', extra_compile_args=['-std=c++11']),
                 Extension('graph', sources=['graph.pyx', './src/lib/graph.cpp'], language='c++', extra_compile_args=['-std=c++11']),
                 Extension('maxcut_env', sources=['maxcut_env.pyx', './src/lib/maxcut_env.cpp', './src/lib/graph.cpp'], language='c++', extra_compile_args=['-std=c++11']),
                 Extension('nstep_replay_mem', sources=['nstep_replay_mem.pyx', './src/lib/nstep_replay_mem.cpp', './src/lib/graph.cpp', './src/lib/maxcut_env.cpp'], language='c++', extra_compile_args=['-std=c++11']),
                 Extension('graph_struct', sources=['graph_struct.pyx', './src/lib/graph_struct.cpp'], language='c++', extra_compile_args=['-std=c++11']),
                 Extension('mcstep', sources=['mcstep.pyx', './src/lib/mcstep.cpp','./src/lib/graph.cpp'], language='c++', extra_compile_args=['-std=c++11']),
                 # Extension('SGRL', sources=['SGRL.py'], language='c++')
                 ])
