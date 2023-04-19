from datetime import datetime
from pathlib import Path
from decorator import decorator

from .io import load_ops


@decorator
def updates_flexilims(func, name_source=None, *args, **kwargs):
    """Updates flexilims when running the function"""
    # find if we should use flexilims
    data_path = args[0]  # decorator ensures all arguments are passed as positional
    ops = load_ops(data_path)
    if ("use_flexilims" in ops) and ops["use_flexilims"]:
        # get parent from flexilims, must exist
        import flexiznam as flz

        data_path = Path(data_path)
        flm_session = flz.get_flexilims_session(project_id=data_path.parts[0])
        parent_name = "_".join(data_path.parts[1:])
        parent = flz.get_entity(
            name=parent_name, datatype="sample", flexilims_session=flm_session
        )
        if parent is None:
            raise ValueError(f"Could not find parent {parent_name} in flexilims")
        print(f'Using flexilims.\n Parent: {parent["name"]}')
    else:
        parent = None
        print("Not using flexilims")

    # Actually run the function
    value = func(*args, **kwargs)

    # update flexilims if needed
    if parent is not None:
        if name_source is None:
            func_name = func.__name__
        else:
            # find the index of the args whose name is `name_source`
            arg_index = func.__code__.co_varnames.index(name_source)
            func_name = args[arg_index]
        dataset_name = f"{parent_name}_{func_name}"
        flm_attr = dict(ops)
        flz.utils.clean_dictionary_recursively(flm_attr)
        print("Adding dataset to flexilims")
        rep = flz.add_dataset(
            parent_id=parent["id"],
            dataset_type="iss_preprocessing",
            created=datetime.now().strftime("%Y-%m-%d " "%H:%M:%S"),
            path=str(data_path),
            genealogy=list(parent["genealogy"]) + [func_name],
            is_raw="no",
            dataset_name=dataset_name,
            attributes=flm_attr,
            flexilims_session=flm_session,
            conflicts="overwrite",
        )
        print(f"Dataset {rep['name']} added to flexilims with id {rep['id']}")
    return value
