import solstice
import solstice.compat

packages = {"solstice": solstice, "solstice.compat": solstice.compat}


def define_env(env):
    @env.macro
    def list_package_members(package_name: str):
        package = packages[package_name]
        output = "!!! abstract \n"
        output += f"    This is the whole API.  Everything is accessible through the `{package_name}.*` namespace:\n"
        for obj in package.__all__:
            output += "\n"
            output += f"    - `{package_name}.{obj}`\n"
        return output
