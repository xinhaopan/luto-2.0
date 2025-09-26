import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import pylustrator
pylustrator.start()

fig, ax = plt.subplots()
ax.plot([0,1,2], [2,1,3])
ax.set_title("pylustrator demo")
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
plt.figure(1).text(0.0160, 0.6087, 'New Text', transform=plt.figure(1).transFigure, rotation=90.)  # id=plt.figure(1).texts[0].new
#% end: automatic generated code from pylustrator
plt.show()
