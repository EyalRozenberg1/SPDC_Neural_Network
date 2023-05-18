%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Free Space propagation using the free space transfer function(two
% dimensional), according to Saleh
% Function reciveing:        
%        -A: The field amplitude (A) to propagate
%        -x,y : spatial vectors
%        -k : the k-vector of the field
%        -d: The distance to propagate
% **with fresnel approximation       
% The output is the propagated field.
% Using CGS, or MKS, Boyd 2nd eddition       
% Sivan Trajtenberg-Mills, 2013 
%
% Changed in 2016: not containg exp(ikz)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Eout=propagate3(A, x , y, k , d )

%define the fourier vectors
dx=abs(x(2)-x(1)); dy=abs(y(2)-y(1));
[X,Y]=meshgrid(x,y); 
KX = 2*pi*(X./dx)./size(X,2)/dx; %2pi/(dx*L)
KY = 2*pi*(Y./dy)./size(Y,1)/dy; %2pi/(dy*L)

%The Free space transfer function of propagation, using the Fresnel
    %approximation (from "Engineering optics with matlab"/ing-ChungPoon&TaegeunKim):
     H_w=exp(-1i*d*(KX.^2+KY.^2)/(2*k)); 

    %The Free space transfer function of propagation, without the Fresnel
    %approximation, according to Saleh, 2nd eddition, 4.1.9, and remembering
    %that we defined E=Aexp[-ikz], so we must multiply by exp[-ikz] to recieve E from A
    %and also remembering the condition k^2 > KX.^2-KY.^2:
    %if(~flag)
        %H_w=exp(1i*d*sqrt( (k.^2-KX.^2-KY.^2).*(k.^2 > KX.^2-KY.^2) )); %.* exp(-1i*k*d); %
    %end
    
    H_w=ifftshift(H_w);%(inverse fast Fourier transform shift). For matrices, ifftshift(X) swaps the first quadrant with the third and the second quadrant with the fourth.

%Fourier Transform: move to k-space    
    G=(fft2(A));% [in Fourier (spatial frequency) space]. the two-dimensional discrete Fourier transform (DFT) of A.
    
%propoagte in the fourier space    
    F=G.*H_w; 
 
%inverse Fourier Transform: go back to real space
   Eout=(ifft2((F)));% [in real space]. E1 is the two-dimensional INVERSE discrete Fourier transform (DFT) of F1
   
end