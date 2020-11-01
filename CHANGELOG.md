1.2.2
- added kwargs to change GLCM neighbourhood (one per image!)
- switched to column-major tile addressing
- optimized kwargs
- cleaned unnecessary comments throughout all modules

1.1.2
- Optimized kwargs in submodule render.cpu
- changed default rowssave value from 42 to 80
- fixed mistake in libglcmsw.render.cpu.rasterrender description
- updated classes.py
- altered TODO.md

1.1.1
- Switched from list comprehensions to itertools.repeat() in process pool executors

1.1.0
- Added the SlidingWindow class