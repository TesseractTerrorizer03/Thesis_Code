"""Execute notebook in-process with IPython when sandboxed kernels cannot bind sockets.

Uses the same cells, real execution, normal rich outputs, and fail-on-error
semantics. No results are fabricated or inserted from illustrative values.
"""
from pathlib import Path
import os,traceback
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from threadpoolctl import threadpool_limits

class LocalShell(InteractiveShell):
    def enable_gui(self,gui=None):return None

def execute():
    root=Path(__file__).resolve().parent;os.chdir(root)
    path=root/'Kent_FX_Anisotropy_Pilot.ipynb';book=nbformat.read(path,as_version=4)
    shell=LocalShell.instance();shell.run_line_magic('matplotlib','inline')
    number=0
    with threadpool_limits(limits=1):
        for i,cell in enumerate(book.cells):
            if cell.cell_type!='code':continue
            number+=1;cell.outputs=[];cell.execution_count=number
            with capture_output() as captured:
                outcome=shell.run_cell(cell.source,store_history=False)
            if captured.stdout:cell.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
            if captured.stderr:cell.outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
            for output in captured.outputs:
                cell.outputs.append(nbformat.v4.new_output('display_data',data=output.data,metadata=output.metadata))
            nbformat.write(book,path)
            error=outcome.error_before_exec or outcome.error_in_exec
            if error:raise RuntimeError(f'Notebook cell {i} failed: {error}') from error
            print(f'Executed cell {i}',flush=True)
    nbformat.validate(book)
    print(f'Executed {number} code cells; saved {path}',flush=True)

if __name__=='__main__':execute()
