using VirtualPlantLab
import ColorTypes: RGBA # for the color of each mesh
import GLMakie # for 3D rndering (native OpenGL Backend)

turtle = Turtle()
p = Triangle!(turtle; length = 1.0, width = 1.0, colors = rand(RGBA))
render(Mesh(turtle), wireframe = true)
using IJulia
notebook()