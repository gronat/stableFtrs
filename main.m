function main(opt)
	% opt:  'track' for feature tracking
	%		'nn' for nearest neighbour with orig image

	close all;
	if (nargin<1) 	opt = 'track'; end;
	hfov = 90;
	pth = '~/Desktop/barak.jpg';

	% *** Read image, detect features and plot it
	img0 = imread(pth);
	[imh imw foo] = size(img0);
	imtsize(1,:) = size(img0);

	[x0, desc0] = detectFtrs(img0);

	% *** Plot image with ftrs
	imshow(img0/2+100);
        ha = gca();
        plotPoints(ha, x0, 'or');

    % *** Apply incremental tranformations and track features
	alpha = [[0:0.5:12]];% [-0.5:-0.5:-	12]]	;
	track = zeros(3 ,size(x0, 2), length(alpha));
	track(1:2,:,1) = x0;
	track(3,:,1) = 1;

	for (k = 2:length(alpha))
		fprintf('Processing %d of %d\n', k, length(alpha));
		% Transofrm image, detect ftrs and project back
		H  				= hrotation2homography(alpha(k), hfov, imw, imh);
		[img1 offset] 	= rotateCamera(img0, H);
		[x1, desc1] 	= detectFtrs(img1);
		x1off 			= bsxfun(@plus, x1, offset); 
		x1 				= projectPoints(x1off, inv(H));
		
		plotPoints(ha, x1, '.g');
		% Track the feature points
		for (j = 1:size(x0,2))
			switch opt
				case 'nn'	 	
					x = track(1:2, j, 1)	; 	% ftr form the first image
				case 'track' 	
					x = track(1:2, j, k-1);	% ftr from the previous image
			end
			% Update the track
			[point id] = findNN(x, x1);
			track(1:2,j,k) 	= point;
			track(3,j,k) = desc0(j, :)*desc1(id, :)';
		end
		imtsize(k,:) = size(img1); 
			
	end



	%plotTrack(ha, track, opt);
	
	% *** Compute covariance matrix of the track and eigenvalues
	lambda = computerCovEigVals(track);
	r = sqrt(sum(lambda.^2, 2));
	bin = lambda(:,2)<1.0;
	
	sim = squeeze(track(3,:,:));
	simvar = var(sim')';
	bin = simvar < 0.005;
	keyboard

	% plotPoints(ha, x0(:,bin), 'sg');
	% plotData(ha, x0, r);

	
	% *** Plot original ftrs and filtered ftrs
	figure()
		imshow(img0/2+100);
		plotPoints(gca, x0, 'ro');

	figure()
		imshow(img0/2+100);
		plotPoints(gca, x0(:, bin), 'gs')

	% *** Plot eigen values
	figure()
		plot(lambda(:,1), lambda(:,2), '.');

	figure()
		plotHeatmap(gca, x0, x0(:,bin), 15)
			

	keyboard
end

function lambda = computerCovEigVals(track)
	N = size(track,2);
	lambda = zeros(N, 3);
	for (k = 1:N)
		X = squeeze(track(1, k, :));
		Y = squeeze(track(2, k, :));
		S = squeeze(track(3, k, :));
		lambda(k,:) = eig(cov([X Y S]));
	end
end

function [xNN idx] = findNN(x, Y)
	sqrdfr = sum(bsxfun(@minus, Y, x).^2);
	[val idx] = min(sqrdfr);
	xNN = Y(:,idx);
end

function xproj = projectPoints(x, H) 
	xproj = h2a(H*a2h(x));
end


function [x desc] = detectFtrs(img)
	thr = 100;
	blobs = detectSURFFeatures(rgb2gray(img), 'MetricThreshold', thr);
	[desc, validBlobs] = extractFeatures(rgb2gray(img), blobs);
	x = [validBlobs.Location]';
end

function [imgh offset] = rotateCamera(img, H)
	% imgh = rotateImage(img, alpha, hfov)
	%
	% Transforms image by a given homography.
	
	tform = projective2d(H');
  	[imgh meta] = imwarp(img, tform);
  	offset = [min(meta.XWorldLimits) min(meta.YWorldLimits)]'-0.5	;
end

function x = h2h(x)
	z = x(end,:);
	x = bsxfun(@rdivide, x, z);
end

function x = a2h(x)
	x = [x; ones(1,size(x,2))];
end

function x = h2a(x)
	x = h2h(x);
	x = x(1:end-1,:);
end

function plotData(h, x, data)
	hold(h, 'on');
	x = double(x);
	offset = 2; 	% text offset in pexels
	N = length(data);
	for (k=1:N)
		text(x(1,k)+offset, x(2,k)+offset, sprintf('%0.2f', data(k)));
 	end
	hold(h, 'off');	
end

function plotTrack(h, track, opt)
	hold(h, 'on');
	N = size(track,2);
	for (k=1:N)
		x0 = track(1,k,1);
		y0 = track(2,k,1);
		X = squeeze(track(1,k,:));
		Y = squeeze(track(2,k,:));
		switch opt
			case 'nn'
				for (j=1:length(X))
					line([x0 X(j)], [y0 Y(j)], 'Parent', h);
				end
			case 'track'
				line(X, Y, 'Parent', h);
		end
	end
	hold(h, 'off');
end

function plotPoints(h, x, opt);
	if (nargin<3) opt = '.'; end;
	hold(h, 'on');
	plot(x(1,:), x(2,:), opt)
	hold(h, 'off');
end

function plotHeatmap(h, xorig, xfilter, winSize)
	%winSize = 50; % Parsen window size in pixels
	xorig = double(xorig);
	xfilter = double(xfilter);
	coordOrig = ceil(xorig/winSize); 	% ceil to 	avoid zero coordinate
	coordFilter = ceil(xfilter/winSize);
	xmax = max([coordOrig(1,:) coordFilter(1,:)]);
	ymax = max([coordOrig(2,:) coordFilter(2,:)]);
	
	Morig 	= full(sparse(coordOrig(1,:), 	coordOrig(2,:), 1, 		xmax, ymax));
	Mfilter = full(sparse(coordFilter(1,:), coordFilter(2,:), 1, 	xmax, ymax));
	M = Mfilter./Morig;
	imagesc(M'	, 'Parent', h);
	colorbar;
	colormap jet;
	axis ij;
end

function H = pts2homography(pts1, pts2)
%	H is 3*3, H*[pts1(:,i);1] ~ [pts2(:,i);1], H(3,3) = 1
%	the solving method see "projective-Seitz-UWCSE.ppt"
	n = size(pts1,2);
	A = zeros(2*n,9);
	A(1:2:2*n,1:2) = pts1';
	A(1:2:2*n,3) = 1;
	A(2:2:2*n,4:5) = pts1';
	A(2:2:2*n,6) = 1;
	x1 = pts1(1,:)';
	y1 = pts1(2,:)';
	x2 = pts2(1,:)';
	y2 = pts2(2,:)';
	A(1:2:2*n,7) = -x2.*x1;
	A(2:2:2*n,7) = -y2.*x1;
	A(1:2:2*n,8) = -x2.*y1;
	A(2:2:2*n,8) = -y2.*y1;
	A(1:2:2*n,9) = -x2;
	A(2:2:2*n,9) = -y2;

	[evec,~] = eig(A'*A);
	H = reshape(evec(:,1),[3,3])';
	H = H/H(3,3); 
end


