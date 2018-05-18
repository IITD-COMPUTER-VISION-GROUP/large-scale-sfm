#-----------------------------------------------------------------------------
# Top level makefile for Bundler
#
# Bundler: Structure from Motion for Unordered Photo Collections
# Version: 0.4 04/03/2010
#    http://phototour.cs.washington.edu/bundler/
#-----------------------------------------------------------------------------
# Copyright (c) 2008-2010 University of Washington and Noah Snavely
# All Rights Reserved.
#-----------------------------------------------------------------------------

ANN_TARGET = linux-g++-shared

OS = $(shell uname -o)
ifeq ($(OS), Cygwin)
ANN_TARGET = win32-g++-shared
endif

default:
# Make libraries	
	cd lib/matrix; $(MAKE)
# Auxiliary libraries
	cd lib/minpack; $(MAKE)
	cd lib/cblas; $(MAKE)
	cd lib/f2c; $(MAKE)

clean:
	cd lib/matrix; $(MAKE) clean
	cd lib/minpack; $(MAKE) clean
	cd lib/cblas; $(MAKE) clean
	cd lib/f2c; $(MAKE) clean
