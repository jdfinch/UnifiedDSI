
import ezpyzy as ez
import pathlib as pl
import fabric as fab

hosts = dict(
    tebuna='localhost:55555',
    h100='localhost:55556'
)

user = 'jdfinch'
project = pl.Path(f'/local/scratch/{user}/UnifiedDSI/')


def download(machine, path, project_root=str(project), target_path=None):
    project = pl.Path(project_root)
    password = pl.Path('~/.pw/emory').expanduser().read_text().strip()
    credentials = dict(
        user=user,
        connect_kwargs=dict(password=password)
    )
    with fab.Connection(hosts[machine], **credentials) as conn:
        path = pl.Path(path)
        is_folder = conn.run(
            f'test -d {project/path} && echo 1 || echo 0'
        ).stdout.strip() == '1'
        if is_folder:
            tar_file = (path if target_path is None else target_path).with_suffix('.tar.gz')
            conn.run(f'cd {project} && tar -czvf {tar_file} {path}')
            conn.get(f'{project / tar_file}', str(tar_file))
            conn.run(f'rm {project / tar_file}')
            conn.local(f'tar -xzvf {tar_file}')
            conn.local(f'rm {tar_file}')
            print(f'Got {project/path}')
        else:
            conn.get(f'{project/path}', str(path) if target_path is None else target_path)
            print(f'Got {project/path}')



if __name__ == '__main__':

    download('tebuna', 'ex/CharismaticLehon_tebu')