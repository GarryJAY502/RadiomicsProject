import os
from typing import List, Optional

def subdirs(folder: str, join: bool = True, prefix: Optional[str] = None, suffix: Optional[str] = None, sort: bool = True) -> List[str]:
    """
    Returns a list of subdirectories in a given folder, optionally filtering by prefix and suffix,
    and optionally sorting the results. Uses os.scandir for efficient directory traversal.

    Parameters:
    - folder: Path to the folder to list subdirectories from.
    - join: Whether to return full paths to subdirectories (if True) or just directory names (if False).
    - prefix: Only include subdirectories that start with this prefix (if provided).
    - suffix: Only include subdirectories that end with this suffix (if provided).
    - sort: Whether to sort the list of subdirectories alphabetically.

    Returns:
    - List of subdirectory paths (or names) meeting the specified criteria.
    """
    
    subdirectories = []
    with os.scandir(folder) as entries:
        for entry in entries:
            if entry.is_dir() and \
                (prefix is None or entry.name.startswith(prefix)) and \
                (suffix is None or entry.name.endswith(suffix)):
                    dir_path = entry.path if join else entry.name
                    subdirectories.append(dir_path)
    
    if sort:
        subdirectories.sort()
    
    return subdirectories

def subfiles(folder: str, join: bool = True, prefix: Optional[str] = None, suffix: Optional[str] = None, sort: bool = True) -> List[str]:
    """
    Returns a list of files in a given folder, optionally filtering by prefix and suffix,
    and optionally sorting the results. Uses os.scandir for efficient directory traversal,
    making it suitable for network drives.

    Parameters:
    - folder: Path to the folder to list files from.
    - join: Whether to return full file paths (if True) or just file names (if False).
    - prefix: Only include files that start with this prefix (if provided).
    - suffix: Only include files that end with this suffix (if provided).
    - sort: Whether to sort the list of files alphabetically.

    Returns:
    - List of file paths (or names) meeting the specified criteria.
    """
    files = []
    with os.scandir(folder) as entries:
        for entry in entries:
            if entry.is_file() and \
               (prefix is None or entry.name.startswith(prefix)) and \
               (suffix is None or entry.name.endswith(suffix)):
                file_path = entry.path if join else entry.name
                files.append(file_path)

    if sort:
        files.sort()

    return files

def maybe_mkdir_p(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)
