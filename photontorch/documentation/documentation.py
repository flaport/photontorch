"""
# Documentation

Custom scripts to generate the photontorch documentation

You can generate and view the documentation by typing [only with python 3]

```
python -m photontorch.documentation
```
"""

## Imports

import os
import re
import pydoc
import shutil


## Parse Module


def parse(module):
    """ Parse the folder tree of a module to find all submodules

    Args:
        module (str): The name of the module to parse

    Yields:
        str: The next submodule

    """
    for root, _, files in os.walk(module):
        root = root.replace("\\", "/").split("/")
        skip = True
        for subdir in root:
            if subdir == "tests":
                break
            if not "__init__.py" in files:  # check if it is a python module
                break
            if subdir == "__pycache__":  # check if folder is pycache
                break
            if subdir[0] == ".":  # check if folder is hidden
                break
        else:
            skip = False
        if not skip:
            root = ".".join(root)
            yield root  # yield submodule folder name
            for file in files:
                file, ext = os.path.splitext(file)
                if file[0] != "." and ext == ".py":
                    yield root + "." + file  # yield submodule file name


## Write Documentation


def writedoc(submodule, folder="."):
    """Write the documentation for a submodule.

    The documentation will be generated from the docstrings.

    Note:
        This function acts as a replacement for pydoc.writedoc with some custom features added.

    Args:
        submodule (str): The name of the submodule to create documentation for

    """
    obj, name = pydoc.resolve(submodule, False)
    page = pydoc.html.page(pydoc.describe(obj), pydoc.html.document(obj, name))

    # Space
    space = r"&nbsp;"

    # Enable math:
    page = page.replace(
        "</head>",
        """
    <script type="text/javascript" async
    src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.2/MathJax.js?config=TeX-MML-AM_CHTML">
    </script>
    </head>
    """,
    )

    # some nasty string manipulations ahead...

    # Change headers:
    for i in range(4, 0, -1):
        for heading in re.findall("\n" + r"#" * i + space + r"(.+?)<br>", page):
            page = page.replace(
                r"#" * i + space + heading + r"<br>",
                r"<h%i>" % i + heading + r"</h%i>" % i,
            )
    for i in range(4, 0, -1):
        for heading in re.findall(r">" + r"#" * i + space + r"(.+?)<br>", page):
            page = page.replace(
                r"#" * i + space + heading + r"<br>",
                r"<h%i>" % i + heading + r"</h%i>" % i,
            )

    # Change latex code blocks:
    for code in re.findall(r"```math<br>(.+?)```", page, re.DOTALL):
        new_code = code.replace(space, " ").replace(r"<br>", " ")
        new_code = new_code.strip()
        if new_code.endswith(r"<br>"):
            new_code = new_code[:-4]
        new_code = new_code.replace(r"<br>", r"\\")
        page = page.replace(
            r"```math<br>" + code + r"```", r"<br>$$" + new_code + r"$$"
        )

    # Change code blocks:
    for code in re.findall(r"```<br>(.+?)```", page, re.DOTALL):
        page = page.replace(r"```<br>" + code + r"```", r"<tt><br>" + code + r"</tt>")

    # Change inline code:
    for code in re.findall(r"`(.+?)`", page):
        page = page.replace(r"`" + code + r"`", r"<i><tt>" + code + r"</tt></i>")

    # Bullets
    page = page.replace(2 * space + "*" + space, 2 * space + r"<li>" + space)

    # Markdown like URLs:
    possible_urls = re.findall(r"\]\((.+?)\)", page, re.DOTALL)
    possible_urlnames = re.findall(r"\[(.+?)\]\(", page, re.DOTALL)
    for url in possible_urls:
        for urlname in possible_urlnames:
            new_url = url.strip()
            # Clean up URL
            if url[:7] == r"<a href":
                new_url = re.findall(r'"(.+?)"', new_url)[0]
            # Relative URLs:
            if new_url[:7] != r"http://" and new_url[:8] != r"https://":
                # Module URL:
                if submodule not in new_url:
                    if new_url[0] != ".":
                        new_url = "." + new_url
                    new_url = submodule + new_url + r".html"
                # Check if module exists
                if not os.path.exists(new_url):
                    new_url = new_url.replace(r".__init__", r"")
                # Check for classes:
                new_url_list = new_url.split(r".")
                if new_url_list[-2][0] in r"ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                    new_url = (
                        ".".join(new_url_list[:-2]) + r".html" + r"#" + new_url_list[-2]
                    )

            # Replace URL:
            page = page.replace(
                r"[%s](%s)" % (urlname, url),
                r'<a href="%s">%s</a>' % (new_url, urlname),
            )

    # check for URLs related to dependencies of PhotonTorch
    # and replace them with a google "I'm Feeling Lucky" search.
    possible_urls = re.findall(r'<a href="(.+?)"', page, re.DOTALL)
    for url in possible_urls:
        if not url == r"." and not r"photontorch" in url and not url[0] == r"#":
            # replace url by a google I'm feeling lucky search:
            search_query = r"+".join(
                (
                    [r"python"]
                    + url.replace(r"#", r".").replace(r".html", r"").split(r".")
                )[::-1]
            )
            new_url = (
                'http://www.google.com/webhp?#q=%s&btnI=I" target="_blank'
                % search_query
            )
            page = page.replace(url, new_url)

    # somehow, this cannot be prevented. TODO: find the source of this weird URL.
    page = page.replace(
        r'photonhttp://www.google.com/webhp?#q=torch+python&btnI=I" target="_blank',
        r"photontorch.html",
    )

    filename = os.path.abspath(os.path.join(folder, name + ".html"))
    with open(filename, "w", encoding="utf-8") as file:
        file.write(page)


################
## Write Docs ##
################


def writedocs(module):
    """Write all the documentation for a module.

    The documentation will be generated from the docstrings.

    Note:
        This function replaces pydoc.writedocs.

    Args:
        module (str): The name of the module to create documentation for

    """
    for submodule in parse(module):
        pydoc.writedoc(submodule)


##########
## main ##
##########


def main():
    """ generate documentation """
    # go to folder containing photontorch
    cwd = os.path.abspath(os.getcwd())
    output_folder = os.path.join(cwd, "photontorch_documentation")
    photontorch_folder = os.path.dirname(os.path.abspath(__file__ + "/../.."))
    os.chdir(photontorch_folder)

    # Parse submodules of photontorch and replace readme.md with docstring of __init__.py files:
    for submodule in parse("photontorch"):
        if submodule == "photontorch.__init__":
            continue  # the root readme.md should not be generated
        submodule_list = submodule.split(".")
        if submodule_list[-1] == "__init__":
            print(submodule)
            init_module, _ = pydoc.resolve(submodule)
            dirname = os.path.abspath(os.path.join(*submodule_list[:-1]))
            readme = os.path.join(dirname, "readme.md")
            with open(readme, "w", encoding="utf-8") as file:
                file.write(
                    "[comment]: # (This is and automatically generated readme file)\n"
                )
                file.write(
                    "[comment]: # (To edit this file, edit the docstring in the __init__.py file)\n"
                )
                file.write(
                    "[comment]: # (And run the documentation: python -m photontorch.documentation)\n"
                )
                file.write(init_module.__doc__)

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    # Parse submodule of photontorch to write html documentation
    for submodule in parse("photontorch"):
        # Write documentation for submodule
        try:
            writedoc(submodule, folder=output_folder)
            print(submodule)
        except Exception as e:
            if submodule.split(".")[-1] == "__main__":
                continue
            print(submodule + "\tFAILED")
            print(e)
            continue

    # view documentation
    import webbrowser

    filename = os.path.join(output_folder, "photontorch.html").replace("\\", "/")
    webbrowser.open("file:///" + filename)
