# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.12

# Default target executed when no arguments are given to make.
default_target: all

.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /sw/eb/sw/CMake/3.12.1/bin/cmake

# The command to remove a file.
RM = /sw/eb/sw/CMake/3.12.1/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /scratch/user/emily.wax/csce435/final_project/14-peas-in-a-pod

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /scratch/user/emily.wax/csce435/final_project/14-peas-in-a-pod

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/sw/eb/sw/CMake/3.12.1/bin/cmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache

.PHONY : rebuild_cache/fast

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake cache editor..."
	/sw/eb/sw/CMake/3.12.1/bin/ccmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache

.PHONY : edit_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /scratch/user/emily.wax/csce435/final_project/14-peas-in-a-pod/CMakeFiles /scratch/user/emily.wax/csce435/final_project/14-peas-in-a-pod/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /scratch/user/emily.wax/csce435/final_project/14-peas-in-a-pod/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean

.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named data_creation

# Build rule for target.
data_creation: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 data_creation
.PHONY : data_creation

# fast build rule for target.
data_creation/fast:
	$(MAKE) -f CMakeFiles/data_creation.dir/build.make CMakeFiles/data_creation.dir/build
.PHONY : data_creation/fast

data_creation.o: data_creation.cpp.o

.PHONY : data_creation.o

# target to build an object file
data_creation.cpp.o:
	$(MAKE) -f CMakeFiles/data_creation.dir/build.make CMakeFiles/data_creation.dir/data_creation.cpp.o
.PHONY : data_creation.cpp.o

data_creation.i: data_creation.cpp.i

.PHONY : data_creation.i

# target to preprocess a source file
data_creation.cpp.i:
	$(MAKE) -f CMakeFiles/data_creation.dir/build.make CMakeFiles/data_creation.dir/data_creation.cpp.i
.PHONY : data_creation.cpp.i

data_creation.s: data_creation.cpp.s

.PHONY : data_creation.s

# target to generate assembly for a file
data_creation.cpp.s:
	$(MAKE) -f CMakeFiles/data_creation.dir/build.make CMakeFiles/data_creation.dir/data_creation.cpp.s
.PHONY : data_creation.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... rebuild_cache"
	@echo "... edit_cache"
	@echo "... data_creation"
	@echo "... data_creation.o"
	@echo "... data_creation.i"
	@echo "... data_creation.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

