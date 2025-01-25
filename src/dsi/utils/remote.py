
import ezpyzy as ez
import pathlib as pl
import fabric as fab

hosts = dict(
    tebuna='localhost:55555',
    h100='localhost:55556'
)

user = 'jdfinch'
prefix = pl.Path(f'/local/scratch')


def download(machine, path, project=f'{user}/UnifiedDSI', target_path=None):
    project = pl.Path(prefix)/project
    password = pl.Path('~/.pw/emory').expanduser().read_text().strip()
    credentials = dict(
        user=user,
        connect_kwargs=dict(password=password))
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


class SSH:

    def __init__(self, machine='tebuna', project=f"{user}/UnifiedDSI", user=user):
        self.project = pl.Path(prefix)/project
        password = pl.Path('~/.pw/emory').expanduser().read_text().strip()
        credentials = dict(
            user=user,
            connect_kwargs=dict(password=password))
        self.connection_settings = fab.Connection(hosts[machine], **credentials)
        self.connection = None

    def __enter__(self):
        self.connection = self.connection_settings.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.connection.__exit__(exc_type, exc_val, exc_tb)
        self.connection = None

    def run(self, command, **kwargs):
        return self.connection.run(f"cd {self.project} && {command}", **kwargs)

    def local(self, command, **kwargs):
        return self.connection.local(command, **kwargs)

    def download(self, path, target_path=None):
        path = pl.Path(path)
        is_folder = self.connection.run(
            f'test -d {self.project/path} && echo 1 || echo 0'
        ).stdout.strip() == '1'
        if is_folder:
            tar_file = (path if target_path is None else target_path).with_suffix('.tar.gz')
            self.connection.run(f'cd {self.project} && tar -czvf {tar_file} {path}')
            self.connection.get(f'{self.project / tar_file}', str(tar_file))
            self.connection.run(f'rm {self.project / tar_file}')
            self.connection.local(f'tar -xzvf {tar_file}')
            self.connection.local(f'rm {tar_file}')
            print(f'Got {self.project/path}')
        else:
            self.connection.get(f'{self.project/path}', str(path) if target_path is None else target_path)
            print(f'Got {self.project/path}')


if __name__ == '__main__':

    download('tebuna', 'ex/CharismaticLehon_tebu')