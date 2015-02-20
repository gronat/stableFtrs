function H = hrotation2homography(alpha, hfov, imw, imh)
	% H = hrotation2homography(alpha, hfov, imw, imh)
	%
	% Computes homography between two horizontally
	% rotated cameras. Assumes that there is no skew,
	% that the aspec ratio is equal to one and that
	% camera centers are aligned.
	% Homography corresponds to a horizontal camera reotation
	% along vertical y-axis.
	% 
	% see Pajdla, T. Elements of Geometry, page 56

	fx = (imw/2) / tan(hfov*pi/180/2);
	fy = fx;
	ux = round(imw/2);
	uy = round(imh/2);
	K = [ 	fx 	0 	ux;
			0 	fy 	uy;
			0 	0	1 	];
	
	R = [ 	1 	0	0;
			0 	1 	0;
			0	0	1 	];

	Ry = [ 	cos(alpha/180*pi) 	0 	sin(alpha/180*pi);
			0			1 		0;
			-sin(alpha/180*pi) 0 	cos(alpha/180*pi)	]; 

	H = K*Ry*R'*inv(K);		% Pajdla, T. Elements of Geometry, page 56
	H = H./H(3,3);
end