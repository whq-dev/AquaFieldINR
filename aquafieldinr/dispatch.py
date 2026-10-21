
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from .runtime import resolve_device

INPUT_FLAGS={"--data","--config","--checkpoint","--queries","--candidates","--concentration-candidates","--optical-candidates"}
OUTPUT_FLAGS={"--output","--draws-output","--output-dir"}
COMMANDS={"train","predict","regional","validate","select","nested-validate"}


def dispatch(manifest,devices,output_dir):
    source=Path(manifest).resolve()
    document=json.loads(source.read_text(encoding="utf-8"))
    if set(document)!={"jobs"} or not document["jobs"]:raise ValueError('E_DISPATCH_019')
    resolved=[str(resolve_device(device)) for device in devices]
    if not resolved:raise ValueError("At least one execution device is required.")
    gpu=[device for device in resolved if device!="cpu"]
    if len(set(gpu))!=len(gpu):raise ValueError("CUDA devices must be unique; one process occupies each GPU.")
    destination=Path(output_dir).resolve()
    if destination.exists():raise FileExistsError('E_DISPATCH_025')
    pending=[];seen=set();outputs=set()
    for job in document["jobs"]:
        if set(job)!={"id","args"} or not isinstance(job["id"],str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*",job["id"]) or job["id"] in seen:
            raise ValueError('E_DISPATCH_029')
        seen.add(job["id"])
        arguments=job["args"]
        if not isinstance(arguments,list) or not arguments or not all(isinstance(arg,str) for arg in arguments) or arguments[0] not in COMMANDS:
            raise ValueError('E_DISPATCH_033')
        if "--device" in arguments or any(arg.startswith("--device=") for arg in arguments):raise ValueError("Dispatch assigns devices; omit --device from job args.")
        job_dir=destination/job["id"];arguments=list(arguments);has_output=False;path_flags=set()
        for index,arg in enumerate(arguments):
            if any(arg.startswith(flag+"=") for flag in INPUT_FLAGS|OUTPUT_FLAGS):raise ValueError('E_DISPATCH_037')
            if arg in INPUT_FLAGS|OUTPUT_FLAGS:
                if arg in path_flags:raise ValueError('E_DISPATCH_039')
                path_flags.add(arg)
                if index+1>=len(arguments):raise ValueError('E_DISPATCH_041')
                path_text=arguments[index+1].replace("{job_dir}",str(job_dir))
                path=Path(path_text)
                if not path.is_absolute():path=source.parent/path
                path=path.resolve()
                if arg in OUTPUT_FLAGS:
                    has_output=True
                    if not path.is_relative_to(job_dir) or path==job_dir or path in outputs:raise ValueError('E_DISPATCH_048')
                    outputs.add(path)
                elif not path.is_file():raise FileNotFoundError(path)
                arguments[index+1]=str(path)
        if not has_output:raise ValueError('E_DISPATCH_052')
        pending.append({"id":job["id"],"args":arguments,"directory":job_dir})
    destination.mkdir(parents=True)
    running={};results=[]
    try:
        while pending or running:
            for slot,device in enumerate(resolved):
                if slot in running or not pending:continue
                job=pending.pop(0);job["directory"].mkdir()
                log=(job["directory"]/"execution.log").open("wb")
                command=[sys.executable,"-m","aquafieldinr",*job["args"],"--device",device]
                environment=os.environ.copy()
                package_root=str(Path(__file__).resolve().parents[1])
                environment["PYTHONPATH"]=package_root+(os.pathsep+environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
                try:process=subprocess.Popen(command,cwd=source.parent,env=environment,stdout=log,stderr=subprocess.STDOUT)
                except BaseException:
                    log.close();results.append({"id":job["id"],"device":device,"exit_code":None,"status":"launch_failed","args":job["args"]})
                    raise
                running[slot]=(process,log,job,device,time.perf_counter())
            for slot,(process,log,job,device,started) in list(running.items()):
                code=process.poll()
                if code is None:continue
                log.close();results.append({"id":job["id"],"device":device,"exit_code":code,"elapsed_seconds":time.perf_counter()-started,"args":job["args"]});del running[slot]
                (destination/"dispatch_result.json").write_text(json.dumps({"jobs":results,"status":"running" if pending or running else "passed" if all(r["exit_code"]==0 for r in results) else "failed"},indent=2),encoding="utf-8")
            if running:time.sleep(.05)
    except BaseException as error:
        for process,log,job,device,started in running.values():
            process.terminate();code=process.wait();log.close()
            results.append({"id":job["id"],"device":device,"exit_code":code,"status":"interrupted","elapsed_seconds":time.perf_counter()-started,"args":job["args"]})
        (destination/"dispatch_result.json").write_text(json.dumps({"jobs":results,"status":"failed","error":str(error),"pending_jobs":[job["id"] for job in pending]},indent=2),encoding="utf-8")
        raise
    if any(row["exit_code"]!=0 for row in results):raise RuntimeError('E_DISPATCH_083')
    return results
