"""Execute real notebook cells and capture their outputs without a kernel socket."""
from pathlib import Path
import os,sys
import nbformat
from IPython.utils.capture import capture_output
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from execute_notebook import LocalShell

def main():
    os.chdir(ROOT);path=ROOT/'Kent_Equities_Anisotropy_Pilot.ipynb'
    book=nbformat.read(path,as_version=4);shell=LocalShell.instance();shell.run_line_magic('matplotlib','inline')
    number=0
    with threadpool_limits(limits=1):
        for cell in book.cells:
            if cell.cell_type!='code':continue
            number+=1;cell.execution_count=number;cell.outputs=[]
            with capture_output() as cap:result=shell.run_cell(cell.source,store_history=False)
            for name,value in [('stdout',cap.stdout),('stderr',cap.stderr)]:
                if value:cell.outputs.append(nbformat.v4.new_output('stream',name=name,text=value))
            for item in cap.outputs:cell.outputs.append(nbformat.v4.new_output('display_data',data=item.data,metadata=item.metadata))
            nbformat.write(book,path)
            if result.error_in_exec or result.error_before_exec:raise RuntimeError(result.error_in_exec or result.error_before_exec)
            print('Executed code cell',number,flush=True)
    nbformat.validate(book)

if __name__=='__main__':main()
