#!/usr/bin/python
import numpy as np
from scipy import ndimage
from scipy.ndimage import measurements
import math
from numpy import matlib, dtype
from numpy.linalg import eig, inv
from numpy import fft

def get_line(x1, y1, x2, y2):
	x1=int(x1)
	y1=int(y1)
	x2=int(x2)
	y2=int(y2)
	points = []
    	issteep = abs(y2-y1) > abs(x2-x1)
   	if issteep:
        	x1, y1 = y1, x1
        	x2, y2 = y2, x2
    	rev = False
    	if x1 > x2:
        	x1, x2 = x2, x1
        	y1, y2 = y2, y1
        	rev = True
    	deltax = x2 - x1
    	deltay = abs(y2-y1)
    	error = int(deltax / 2)
    	y = y1
    	ystep = None
    	if y1 < y2:
        	ystep = 1
    	else:
        	ystep = -1
    	for x in range(x1, x2 + 1):
        	if issteep:
            		points.append((y, x))
       		else:
            		points.append((x, y))
        	error -= deltay
        	if error < 0:
            		y += ystep
            		error += deltax
    # Reverse the list if the coordinates were reversed
    	if rev:
        	points.reverse()
    	return points

def fit_ellipse(x,y):
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:,n]
    return a

def ellipse_center(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    return np.array([x0,y0])

def ellipse_angle_of_rotation( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    return 0.5*np.arctan(2*b/(a-c))

def ellipse_axis_length( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    return np.array([res1, res2])

def get_distance(x1, y1, x2, y2):
	return math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))

def get_outline_mask(p, rows, cols):
	s1=p.split('(')
	x=[]
	y=[]
	for i in xrange(1, len(s1)):
		s2=s1[i].split(')')
		s3=s2[0].split(',')
		x.append(float(s3[0]))
		y.append(float(s3[1]))
		
	
	for c in xrange(len(x)):
		x[c]=round(x[c])
		y[c]=round(y[c])
		if x[c]<0:
			x[c]=0
		if y[c]<0:
			y[c]=0
		if x[c]>=cols:
			x[c]=cols-1
		if y[c]>=rows:
			y[c]=rows-1

		  	
	x_add=[]
	y_add=[]


	for i in xrange(len(x)-1):
		if get_distance(x[i], y[i], x[i+1], y[i+1])>1.4142:
			line = get_line(x[i], y[i], x[i+1], y[i+1])
			if len(line)>2:
				for j in xrange(1,len(line)):
					x_add.append(line[j][0])
					y_add.append(line[j][1])

	if get_distance(x[0], y[0], x[len(x)-1], y[len(y)-1])>1.4142:
		line = get_line(x[0], y[0], x[len(x)-1], y[len(x)-1])
		x2 = [line[i][0] for i in xrange(1, len(line)-1)]
		y2 = [line[i][1] for i in xrange(1, len(line)-1)]
		for i in xrange(len(x2)):
			x.append(x2[i])
			y.append(y2[i])

	
	img=np.zeros((rows, cols))
	
	for i in xrange(len(x)):
		img[y[i], x[i]]=1
	
	for i in xrange(len(x_add)):
		img[y_add[i], x_add[i]]=1
	
	for i in xrange(rows):
		for j in xrange(cols):
			
			if img[i, j]==0:
				neighors1=[]
				neighors2=[]
				neighors3=[]
				neighors1.append((i-1, j-1))
				neighors1.append((i-1, j))
				neighors1.append((i-1, j+1))
				neighors3.append((i, j-1))
				neighors3.append((i, j+1))
				neighors2.append((i+1, j-1))
				neighors2.append((i+1, j))
				neighors2.append((i+1, j+1))
				flag1=0
				flag2=0
				flag3=0
				for n in neighors1:
					k=n[0]
					l=n[1]
				
					if k>=0 and k<rows and l>=0 and l<cols:
						if img[k, l]==1:
							#print flag
							flag1=flag1+1
				for n in neighors2:
					k=n[0]
					l=n[1]
					if k>=0 and k<rows and l>=0 and l<cols:
						if img[k, l]==1:
							flag2=flag2+1
				for n in neighors3:
					k=n[0]
					l=n[1]
					if k>=0 and k<rows and l>=0 and l<cols:
						if img[k, l]==1:
							flag3=flag3+1

				if (flag1>=1 and flag2>=1) or flag3==2:
					img[i, j] = 1					


		
	return img


def rasterize(img):
	return ndimage.binary_fill_holes(img).astype(np.uint8)
def mean_pixel_value(img, mask):
        assert(np.array_equal(img.shape, mask.shape)==True)

        newimg=np.zeros(img.shape)

        for i in xrange(img.shape[0]):
                for j in xrange(img.shape[1]):
                        if mask[i, j]==1:
                                newimg[i,j]=img[i,j]

        return np.mean(newimg)

def std_dev_pixel_value(img, mask):
        assert(np.array_equal(img.shape, mask.shape)==True)

        newimg=np.zeros(img.shape)

        for i in xrange(img.shape[0]):
                for j in xrange(img.shape[1]):
                        if mask[i, j]==1:
                                newimg[i, j]=img[i, j]
        return np.std(newimg)

def effective_diameter(mask):
        area=np.sum(mask)
        d=math.sqrt(area/math.pi)
        return d

def degree_of_circularity(mask):
	centroids = ndimage.measurements.center_of_mass(mask)
	x = 0
	y = 0
	if type(centroids)==type([]):
		x = np.mean(np.array([pt[1] for pt in centroids]))
		y = np.mean(np.array([pt[0] for pt in centroids]))
	else:
		x = centroids[1]
		y = centroids[0]
	
	
	diameter = effective_diameter(mask)
#	MatX = np.matlib.repmat(np.array([e-x for e in  xrange(mask.shape[1])]), mask.shape[0], 1)
#	MatY = np.matlib.repmat(np.transpose(np.array([e-y for e in xrange(mask.shape[0])])), 1, mask.shape[1])
	XY = np.empty(mask.shape, object)

	for i in xrange(mask.shape[0]):
		for j in xrange(mask.shape[1]):
			XY[i, j] = (j-x, i-y)
	
	Z=np.zeros(mask.shape)
		
	for i in xrange(Z.shape[0]):
		for j in xrange(Z.shape[1]):
			Z[i, j] = XY[i, j][0]**2 + XY[i, j][1]**2
			
	C=np.zeros(mask.shape)
	
	for i in xrange(Z.shape[0]):
		for j in xrange(Z.shape[1]):
			if Z[i, j] <= (diameter*diameter):
				C[i, j] = 1
 	
	C = rasterize(C).astype(np.uint)

	Cm = np.zeros(mask.shape)
	nodule_area = np.sum(mask)
	for i in xrange(mask.shape[0]):
		for j in xrange(mask.shape[1]):
			if mask[i, j]==1 and C[i, j]==1:
				Cm[i, j] = 1       
	overlapping_nodule_area = np.sum(Cm)
	
	return overlapping_nodule_area/nodule_area

def degree_of_ellipticity(mask):
	pts = np.nonzero(mask)
	#print pts	
	a  = fit_ellipse(np.array([pts[1][i] for i in xrange(pts[1].shape[0])]), np.array([pts[0][i] for i in xrange(pts[0].shape[0])]))
	center = ellipse_center(a)
	alpha = ellipse_angle_of_rotation(a)
	axes = ellipse_axis_length(a)
	
	t = np.linspace(0, 2*math.pi, mask.shape[0])
	Q = np.array([[math.cos(alpha), -math.sin(alpha)], [math.sin(alpha), math.cos(alpha)]])
	
	tarr = np.empty((1, 2)) 
	
	for i in xrange(t.shape[0]):
		tarr = np.vstack((tarr, [axes[0]*math.cos(t[i]), axes[1]*math.sin(t[i])]))
	
	tarr =np.transpose(tarr)
	
	z = np.empty((1, 2))

        for i in xrange(t.shape[0]):
                z = np.vstack((z, [center[0], center[1]]))
	
	z=np.transpose(z)
	
	X = np.transpose(np.dot(Q, tarr) + z).astype(int)

	E = np.zeros(mask.shape)
	
	for i in xrange(X.shape[0]):
		if X[i, 1]>=0 and X[i, 1]<mask.shape[0] and X[i, 0]>=0 and X[i, 0]<mask.shape[1]:
			E[X[i, 1], X[i, 0]]=1

	E = rasterize(E).astype(np.uint)
	
	Em = np.zeros(mask.shape)
	nodule_area = np.sum(mask)
	for i in xrange(mask.shape[0]):
		for j in xrange(mask.shape[1]):
			if mask[i, j]==1 and E[i, j]==1:
				Em[i, j] = 1       
	overlapping_nodule_area = np.sum(Em)
	
    	return overlapping_nodule_area/nodule_area
	
def marginal_irregularity(mask):

        pts = np.nonzero(mask)
        #print pts
        a  = fit_ellipse(np.array([pts[1][i] for i in xrange(pts[1].shape[0])]), np.array([pts[0][i] for i in xrange(pts[0].shape[0])]))
        center = ellipse_center(a)
        alpha = ellipse_angle_of_rotation(a)
        axes = ellipse_axis_length(a)

        t = np.linspace(0, 2*math.pi, mask.shape[0])
        Q = np.array([[math.cos(alpha), -math.sin(alpha)], [math.sin(alpha), math.cos(alpha)]])

        tarr = np.empty((1, 2))

        for i in xrange(t.shape[0]):
                tarr = np.vstack((tarr, [axes[0]*math.cos(t[i]), axes[1]*math.sin(t[i])]))

        tarr =np.transpose(tarr)

        z = np.empty((1, 2))

        for i in xrange(t.shape[0]):
                z = np.vstack((z, [center[0], center[1]]))

        z=np.transpose(z)

        X = np.transpose(np.dot(Q, tarr) + z).astype(int)

        E = np.zeros(mask.shape)

        for i in xrange(X.shape[0]):
                if X[i, 1]>=0 and X[i, 1]<mask.shape[0] and X[i, 0]>=0 and X[i, 0]<mask.shape[1]:
                        E[X[i, 1], X[i, 0]] = 1

        E = E.astype(np.uint8)
        pts1 = np.nonzero(E)
        degree_of_irregularity = 1 - pts[0].shape[0]/pts1[0].shape[0]

        pt_e = np.array([[pts1[0][i], pts1[1][i]] for i in xrange(pts1[0].shape[0])])
        d = np.zeros(pts[1].shape[0])
        for i in xrange(pts[0].shape[0]):
                point = [pts[0][i], pts[1][i]]
                d[i] = np.amin(np.sqrt(np.array(np.sum((pt_e-np.matlib.repmat(point, pts1[0].shape[0], 1))**2,1))))
        ps = np.absolute(np.fft.fft(d)**2)
	
        first_moment = np.sum(ps)/ps.shape[0]
        rmsv = math.sqrt(np.sum((ps - np.mean(ps))**2)/ps.shape[0])

        return first_moment, rmsv, degree_of_irregularity


def border_definition(image, mask):
	n = 1 # should be a odd number
	
	#Eassert(n>=1 and (n/2)!=math.floor(n/2))
	
	# code to handle cases n>1 is needed, but for now we can go without it.

	F = np.gradient(image.astype(np.float64))

	Fm = (F[0] + F[1])/2

	for i in xrange(mask.shape[0]):
		for j in xrange(mask.shape[1]):
			if mask[i, j]!=1:
				Fm[i, j] = 0

	return np.mean(Fm)
		
def radial_gradient_index(image, mask):
	pts = np.nonzero(mask)
	ind = np.arange(mask.shape[0] * mask.shape[1])
	ind = ind.reshape([mask.shape[0], mask.shape[1]])
	I = [ind[pts[0][i], pts[1][i]] for i in xrange(pts[0].shape[0])]
	Cx = np.mean(pts[1])
	Cy = np.mean(pts[0])
	
	F = np.gradient(image.astype(np.float64))
        Fx = []
	Fy = []
	for i in I:
		if i<F[1].shape[0] and i<F[0].shape[0] and i>=0:
			Fx.append(F[1][i])
			Fy.append(F[0][i])

	phi = np.arctan2(np.array(Fy), np.array(Fx))
	xDiff = Cx - pts[1]
	yDiff = Cy - pts[0]
	a = np.arctan2(yDiff, xDiff)
	r = phi - a
	
	RG_nominator = np.sum(np.absolute(np.multiply(np.cos(r), np.sqrt(Fx**2+Fy**2))))
	RG_denominator = np.sum(np.sqrt(Fx**2 + Fy**2))	
	RG = RG_nominator/RG_denominator
	return RG

def tangential_gradient_index(image, mask):
        pts = np.nonzero(mask)
        ind = np.arange(mask.shape[0] * mask.shape[1])
        ind = ind.reshape([mask.shape[0], mask.shape[1]])
        I = [ind[pts[0][i], pts[1][i]] for i in xrange(pts[0].shape[0])]
        Cx = np.mean(pts[1])
        Cy = np.mean(pts[0])

        F = np.gradient(image.astype(np.float64))
        Fx = []
        Fy = []
        for i in I:
                if i<F[1].shape[0] and i<F[0].shape[0] and i>=0:
                        Fx.append(F[1][i])
                        Fy.append(F[0][i])

        phi = np.arctan2(np.array(Fy), np.array(Fx))

        xDiff = Cx - pts[1]
        yDiff = Cy - pts[0]
        a = -np.arctan2(yDiff, xDiff)
        r = phi - a

        TG_nominator = np.sum(np.absolute(np.multiply(np.cos(r), np.sqrt(Fx**2+Fy**2))))
        TG_denominator = np.sum(np.sqrt(Fx**2 + Fy**2))
        TG = TG_nominator/TG_denominator
        return TG

def create_gauss(theta, gaussLength, maskWidth, gaussSigma, maxSigma):
	hwidth = float(float(maskWidth-1)/2)
	x = np.arange(-hwidth, hwidth+1, 1)
	y = np.arange(-hwidth, hwidth+1, 1)
	xn = np.empty((0, maskWidth))
	yn = np.empty((maskWidth, 0))
	x = x.reshape(1, x.shape[0])
	y = y.reshape(y.shape[0], 1)
	for i in xrange(maskWidth):
		xn = xn.vstack((xn, x))
		yn = yn.hstack((yn, y))

	xp = xn * math.cos(theta) - yn * math.sin(theta)
	yp = xn * math.sin(theta) - yn * math.cos(theta)
	
	ind0_y = np.zeros(xp.shape)
	ind0_x = np.zeros(yp.shape)

	for i in xrange(xp.shape[0]):
		for j in xrange(xp.shape[1]):
			if abs(xp[i, j])>float(gaussLength)/2:
				ind0_y[i, j]=1

        for i in xrange(yp.shape[0]):
                for j in xrange(yp.shape[1]):
                        if abs(yp[i, j])>3*math.ceil(maxSigma):
                                ind0_x[i, j]=1

	ind0 = np.logical_or(ind0_x.astype(bool), ind0_y.astype(bool)).astype(int)
	gauss_mask = -np.exp(-0.5 * (xp/gaussSigma)**2)/(math.sqrt(2 * math.pi) * gaussSigma)
	
	for i in xrange(ind0.shape[0]):
		for j in xrange(ind0.shape[1]):
			if ind0[i, j] == 1:
				gauss_mask[i, j] = 0

	#ind1 = np.zeros(ind0.shape)

	ind1 = np.invert(ind0.astype(bool)).asytpe(int)

	mean = np.sum(np.nonzero(gauss_mask))/np.sum(ind1)
	
        for i in xrange(ind1.shape[0]):
                for j in xrange(ind1.shape[1]):
                        if ind1[i, j] == 1:
                                gauss_mask[i, j] = gauss_mask[i, j] - mean

	return gauss_mask, xn
	
		

class gauss:
	length = None
	sigma = None
	maskWidth = None
	theta = None
	templates = None

def create_gaussian_masks(gaussLength, maskWidth, maxSigma, gaussSigma, theta):
	g = gauss()	
 	g.length = gaussLength
	g.sigma = gaussSigma
	g.maskWidth = maskWidth
	g.theta = theta
	nsigma = np.sum(np.array([g.sigma.shape[i] for i in xrange(len(g.sigma.shape))]));
	ntheta = np.sum(np.array([g.sigma.theta[i] for i in xrange(len(g.theta.shape))]));
	templates = np.zeros(g.maskWidth, g.maskWidth, nsigma, ntheta)
	
	for i in xrange(nsigma):
		for j in xrange(ntheta):
			templates[:, :, i, j] = create_gauss(g.theta[j], g.length, g.maskWidth, g.sigma[i], maxSigma)
	g.templates = templates

def line_enhancement_index(image, mask):
	filter_size = 11
	
def calculate_obj_features(image, p):
	outline_mask = get_outline_mask(p, image.shape[0], image.shape[1])
	mask = rasterize(outline_mask)
	
	features = []
	
	f = mean_pixel_value(image, mask)
	features.append(f)
	f = std_dev_pixel_value(image, mask)
	features.append(f)
	f = effective_diameter(mask)
	features.append(f)
	f = degree_of_circularity(mask)
	features.append(f)
	f = degree_of_ellipticity(outline_mask)
	features.append(f)
	f1, f2, f3 = marginal_irregularity(outline_mask)
	features.append(f1)
	features.append(f2)
	features.append(f3)
	f = border_definition(image, outline_mask)
	features.append(f)
#	f = radial_gradient_index(image, outline_mask)	
#	features.append(f)
#	f = tangential_gradient_index(image, outline_mask)

	return features

	
if __name__=='__main__':
	import sys
 
	img=get_outline_mask('(298, 436),(298, 436),(298, 434),(298, 428),(298, 425),(298, 422),(299, 417),(299, 415),(300, 414),(300, 413),(300, 412),(300, 411),(300, 409),(300, 408),(297, 398),(295, 395),(291, 392),(290, 390),(289, 389),(285, 388),(282, 388),(278, 387),(277, 387),(276, 387),(275, 387),(272, 387),(271, 387),(271, 388),(270, 388),(270, 389),(268, 390),(267, 394),(264, 397),(263, 401),(260, 407),(258, 411),(254, 418),(254, 421),(254, 422),(254, 424),(254, 426),(254, 429),(254, 431),(254, 432),(255, 435),(258, 436),(261, 437),(267, 438),(271, 440),(273, 440),(274, 440),(276, 440),(277, 440),(278, 440),(280, 440),(282, 440),(283, 440),(284, 440),(286, 440),(287, 440),(288, 440),(289, 440),(290, 438),(292, 437),(294, 437),(294, 436),(296, 434),(297, 431),(299, 428),(300, 426),(300, 424),(302, 423)', 500, 500)
	#print np.array_equal(img.astype(np.uint8), rasterize(img))
	val = degree_of_circularity(rasterize(img))
	val = degree_of_ellipticity(img) 
	print val
	fm, rmsv, doi = marginal_irregularity(img)
	print border_definition(img, img)
