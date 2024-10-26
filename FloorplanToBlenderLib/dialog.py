from pyfiglet import Figlet
def figlet(text="Floorplan to Blender3d", font="slant"):
    f = Figlet(font=font)
    print(f.renderText(text))


def init():
    print("----- CREATE BLENDER PROJECT FROM FLOORPLAN WITH DIALOG -----")
    print("")


def question(text, default):
    """
    @Param text, question string
    @Param default, possible values
    @Return input
    """
    return input(text + " [default = " + default + "]: ")


