import glob
import os
import getpass


ignore_list = ["index.html"]


def generate_index_html(file_list, title="File Preview", file_list_others=[]):
    """
    creates index.html to show the images in file_list as a gallery
    """
    html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        .gallery {{
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
        }}

        .gallery div {{
            flex-basis: 30%;
            margin: 1%;
            border: 2px solid black;
            box-sizing: border-box;
        }}

        .gallery img {{
            width: 100%;
            height: auto;
            display: block;
        }}

        .gallery a {{
            text-decoration: none;
            color: black;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>

        <div class="gallery">
            {gallery}
        </div>

        <div>
            {others}
        </div>


        
    </div>
</body>
</html>
"""
    gallery_items = []

    for file in file_list:
        if file.endswith(".png"):
            base_name = file[:-4]  # Remove '.png' extension
            pdf_version = base_name + ".pdf"
            additional_link = (
                f"<a href='{pdf_version}' target='_blank' style='color: blue; text-decoration: underline;'>{pdf_version}</a>"
                if pdf_version in file_list
                else ""
            )

            item_html = f"""
            <div>
                <a href="{file}" target="_blank">
                    <img src="{file}" alt="{file}" style="display: block; width: 100%; height: auto;">
                </a>
                <div style="text-align: bottom left; border: 0px;">
                    <a href='{file}' target='_blank' style='color: blue; text-decoration: underline;'>{file}</a><br>
                    {additional_link}
                </div>
            </div>

            """
            gallery_items.append(item_html)

    others = []
    for file in file_list_others:
        if file in ignore_list:
            continue
        others.append(
            f"""
            <a href="{file}" target="_blank" style="color: blue; text-decoration: underline;">{file}</a><br>
    """
        )

    return html_template.format(
        gallery="\n".join(gallery_items), title=title, others="\n".join(others)
    )


def list_files(directory, formats=[".png", ".pdf"]):
    files = [os.path.basename(f) for f in glob.glob(directory + "/*.*")]
    files.sort()
    files_selected = [f for f in files if os.path.splitext(f)[-1] in formats]
    files_others = [f for f in files if f not in files_selected]
    return files_selected, files_others


def generate_html_from_dir(directory):
    files, files_others = list_files(directory, formats=[".png", ".pdf"])
    print("found files: %s" % files)

    html_content = generate_index_html(
        files, title=directory, file_list_others=files_others
    )
    with open(os.path.join(directory, "index.html"), "w") as file:
        file.write(html_content)

    user = getpass.getuser()

    url = show_url(directory, verbose=True)
    # for pathbase, urlbase in cases:
    #     if directory.startswith(pathbase):
    #         url = directory.replace(pathbase, urlbase)
    #         print(
    #             f"""HTML file generated in {directory}: \n{url}/index.html
    #                """
    #         )
    #         break


def show_url(directory, verbose=True):
    import getpass

    user = getpass.getuser()
    cases = [
        (f"/home/{user}/public_html", f"https://user-web.icecube.wisc.edu/~{user}/"),
        (f"/data/user/{user}/", f"https://convey.icecube.wisc.edu/data/user/{user}/"),
        (f"/afs/ifh.de/user/n/{user}/www/", f"https://www-zeuthen.desy.de/~{user}/"),
        (
            f"/lustre/fs23/group/icecube/{user}/www/",
            f"/Users/{user}/clusters/lustre/www/",
        ),
    ]
    for pathbase, urlbase in cases:
        if directory.startswith(pathbase):
            url = directory.replace(pathbase, urlbase)
            if verbose:
                print(f"directory %s --> \n%s/index.html" % (directory, url))
            return url


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate an HTML file to display files in a directory."
    )
    parser.add_argument("--path", type=str, help="Path to the directory", required=True)
    args = parser.parse_args()

    directory = args.path
    generate_html_from_dir(directory)
