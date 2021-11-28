import os

if __name__ == '__main__':
    for i in range(100):
        A = compile(f"""
from nni_exps import main_nni_runner_tt
main_nni_runner_tt.set_seed({i})
main_nni_runner_tt.main()""", "out", 'exec')
        exec(A)
