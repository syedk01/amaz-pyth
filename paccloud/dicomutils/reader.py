#!/usr/bin/python

import vtk
import numpy as np
#from paraview import numpy_support
import dicom

def readwithvtk(file):
	reader=vtk.vtkDICOMImageReader()
	reader.SetFileName(file)
	reader.Update()
	_extent=reader.GetDataExtent()
	ConstPixelDims = [_extent[1]-_extent[0]+1, _extent[3]-_extent[2]+1, _extent[5]-_extent[4]+1]
	ConstPixelSpacing = reader.GetPixelSpacing()
	imageData = reader.GetOutput()
	pointData = imageData.GetPointData()
	assert(pointData.GetNumberOfArrays()==1)
	arrayData = pointData.GetArray(0)
#	ArrayDicom = numpy_support.vtk_to_numpy(arrayData)
#	ArrayDicom = ArrayDicom.reshape(ConstPixelDims, order = 'F')
#	return ArrayDicom
	return arrayData

def read(file):
	ds=dicom.read_file(file)
	img=ds.pixel_array
	return img

if __name__=="__main__":
	import sys
	arr=read(sys.argv[1])
	print type(arr)
	print arr.shape
