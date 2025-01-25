
import dsi.utils.remote as rem

ssh = rem.SSH('tebuna')
with ssh:
    ssh.run(f'./run.sh dsi/results/tabulate.py')
    ssh.download('results')