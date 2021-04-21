import os
import tqdm
import shutil
from .. import haven_utils as hu
from .. import haven_jupyter as hj


def to_dropbox(exp_list, savedir_base, dropbox_path, access_token, zipname):
    """[summary]

    Parameters
    ----------
    exp_list : [type]
        [description]
    savedir_base : [type]
        [description]
    dropbox_path : [type]
        [description]
    access_token : [type]
        [description]
    """
    # zip files
    exp_id_list = [hu.hash_dict(exp_dict) for exp_dict in exp_list]
    src_fname = os.path.join(savedir_base, zipname)
    out_fname = os.path.join(dropbox_path, zipname)
    zipdir(exp_id_list, savedir_base, src_fname)

    upload_file_to_dropbox(src_fname, out_fname, access_token)
    print("saved: https://www.dropbox.com/home/%s" % out_fname)


def upload_file_to_dropbox(src_fname, out_fname, access_token):
    import dropbox

    dbx = dropbox.Dropbox(access_token)
    try:
        dbx.files_delete_v2(out_fname)
    except Exception:
        pass
    # with open(src_fname, 'rb') as f:
    #     dbx.files_upload(f.read(), out_fname)

    upload(access_token=access_token, file_path=src_fname, target_path=out_fname)


def upload(
    access_token,
    file_path,
    target_path,
    timeout=900,
    chunk_size=4 * 1024 * 1024,
):
    import os
    import dropbox
    import tqdm

    dbx = dropbox.Dropbox(access_token, timeout=timeout)
    with open(file_path, "rb") as f:
        file_size = os.path.getsize(file_path)
        chunk_size = 4 * 1024 * 1024
        if file_size <= chunk_size:
            print(dbx.files_upload(f.read(), target_path))
        else:
            with tqdm.tqdm(total=file_size, desc="Uploaded") as pbar:
                upload_session_start_result = dbx.files_upload_session_start(f.read(chunk_size))
                pbar.update(chunk_size)
                cursor = dropbox.files.UploadSessionCursor(
                    session_id=upload_session_start_result.session_id,
                    offset=f.tell(),
                )
                commit = dropbox.files.CommitInfo(path=target_path)
                while f.tell() < file_size:
                    if (file_size - f.tell()) <= chunk_size:
                        print(dbx.files_upload_session_finish(f.read(chunk_size), cursor, commit))
                    else:
                        dbx.files_upload_session_append(
                            f.read(chunk_size),
                            cursor.session_id,
                            cursor.offset,
                        )
                        cursor.offset = f.tell()
                    pbar.update(chunk_size)
    print("uploaded!")


def zipdir(
    exp_id_list,
    savedir_base,
    src_fname,
    add_jupyter=True,
    verbose=1,
    fname_list=None,
    dropbox_path="/shared",
    access_token=None,
):
    import zipfile

    zipf = zipfile.ZipFile(src_fname, "w", zipfile.ZIP_DEFLATED)

    # ziph is zipfile handle
    if add_jupyter:
        abs_path = os.path.join(savedir_base, "results.ipynb")
        hj.create_jupyter(
            fname=abs_path, savedir_base="results/", overwrite=False, print_url=False, create_notebook=True
        )

        rel_path = "results.ipynb"
        zipf.write(abs_path, rel_path)
        os.remove(abs_path)

    n_zipped = 0
    if verbose:
        tqdm_bar = tqdm.tqdm
    else:

        def tqdm_bar(x):
            return x

    fname_all = ["score_list.pkl", "exp_dict.json"]
    if isinstance(fname_list, list):
        fname_all += fname_list

    for exp_id in tqdm_bar(exp_id_list):
        if not os.path.isdir(os.path.join(savedir_base, exp_id)):
            continue

        for fname in fname_all:
            abs_path = os.path.join(savedir_base, exp_id, fname)
            rel_path = os.path.join("results", exp_id, fname)
            if os.path.exists(abs_path):
                zipf.write(abs_path, rel_path)

        n_zipped += 1

    zipf.close()
    if verbose:
        print("Zipped: %d/%d exps in %s" % (n_zipped, len(exp_id_list), src_fname))

    if access_token is not None and access_token != "":
        out_fname = os.path.join(dropbox_path, src_fname)
        upload_file_to_dropbox(src_fname, out_fname, access_token)
        print("saved: https://www.dropbox.com/home/%s" % out_fname)
