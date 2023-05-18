%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PDC, with HG01 crystal
% Units in MKS
% 
% pump wave is K2w: here is k1. resulting idler is E3
% Signal wave E2 is a small gaussian with waist omega02 
%
% based on Noa's code
% Sivan Trajtenberg-Mills, Jan. 2016
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%clear all; close all; pack;

c       = 2.99792458e8;%in meter/sec 
d33     = 16.9e-12/sqrt(1.05)%16.9e-12;%16.9e-12;% in pico-meter/Volt. KTP]
eps0    = 8.854187817e-12; % the vacuum permittivity, in Farad/meter.
I       = @(A,n) 2.*n.*eps0.*c.*abs(A).^2;  
h_bar   = 1.054571800e-34; % Units are m^2 kg / s, taken from http://physics.nist.gov/cgi-bin/cuu/Value?hbar|search_for=planck
simVec=[0,0,0.169,0.175,0.185,0.182,0.174,0.173,0.177,0.177,0.175,0.178,0.184,0.194,0.207,0.228,0.250,0.265,0.277,0.280,0.282]
PumpPvec=[0.0015:0.0001:0.0025];
PumpPvec=[0.0007,0.0011,0.0018,0.002,0.0021,0.0023]*0.9* 9e-3/(1.3e-9)/(pi*(175e-4)^2)*1e-6;
PumpPvec=[0.0007,0.0011,0.0018,0.002,0.0021,0.0023];
PumpPvec=[0.0018,0.002,0.0021];
for p=PumpPvec
%structure arrays
dz=10e-6; dx=2e-6; dy=2e-6; % this was 0.1 um X 0.5 um X 0.5 um
MaxX=700e-6; x=-MaxX:dx:MaxX-dx;
MaxY=700e-6; y=-MaxY:dy:MaxY-dy;
[X,Y] = meshgrid(x,y);
Power2D = @(A,n) sum(sum(I(A,n)))*dx*dy;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Interacting Wavelengths%
lambda_i=1550e-9;%1550e-9;%532e-9;
lambda_p=1064.5e-9;%1064e-9;%3000e-9;
% lambda_p=407e-9;%1064e-9;%3000e-9;
lambda_s=(lambda_i*lambda_p)/(lambda_i+lambda_p);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


T=68.5; %temperature, celsius

omega0_p=175e-6; %pump waist

omega0_i=95e-6; %idler waist  - used if idler is input
omega0_i=95e-6;
omega0_s=100e-6; %signal waist - used if signal is input

%PumpPower = (0.9*100e-6)/(4.5e-9); %LUCE Peak Power 20 kW = 1W * 100 us / 4.5 ns
%PumpPower = (10*1e-6)/(10e-12); %some picosecond laser: Peak Power 1 MW = 10W * 1 us / 10 ps


PumpPower = 0.9*(p * 9e-3)/(1.3e-9); %Alphalas: Peak Power 1 MW = 0.1 W * 10 ms / 1 ns
%For KTP the damage threshold of 10ns pulse is 200MW/cm^2.
%For waist of 300 micron pump the damage is 500smW average power


%pump wave
n_p = nz_KTP_kato(lambda_p*1e6,T);%n_ktp_z(lambda_p*1e6);
w_p = 2*pi*c/lambda_p;
k_p= 2*pi*n_p/lambda_p;
b_p = omega0_p^2*k_p;

%idler wave
n_i = nz_KTP_kato(lambda_i*1e6,T);%n_ktp_z(lambda_i*1e6);
w_i = 2*pi*c/lambda_i;
k_i= 2*pi*n_i/lambda_i;
b_i = omega0_i^2*k_i;

%signal wave
n_s = nz_KTP_kato(lambda_s*1e6,T);%n_ktp_z(lambda_s*1e6);
w_s = 2*pi*c/lambda_s;
k_s = 2*pi*n_s/lambda_s;
b_s = omega0_s^2*k_s;

delta_k=(k_p+k_i-k_s);
Lambda=abs(2*pi/delta_k);



Design = [[20e-3, Lambda/2];[25e-3, 0.13*Lambda];[25e-3, 0.5*Lambda]];
% Design_filenames = {'Raicol_Design1_20mm_uniformDC.mat','Raicol_Design2_20mm_gradient.mat'};
% folder_name = 'C:\Students\Aviv\SG Raicol\Simulation\';


Theta=zeros(1,11);
E_farfield_tot_profile_avg=zeros(1,length(X));
Es_farfield_avg=zeros(size(X));
Ei_farfield_avg=zeros(size(X));
for q=3:3
    MaxZ=Design(q, 1);
    FocusZ = MaxZ / 2;
    Z=0:dz:MaxZ-dz;
    %crystal parameters 
    Poling_period=delta_k;
    W=300e-6;
    Width_index=floor(W/dx);
    PumpOffsetX = 0;

    Dmin=Design(q, 2)/Lambda;
    Gradient_min=sin(pi*Dmin);
    Gradient=zeros(1,length(y));
    
    %Limited gradient: option 1 
    Gradient(end/2-Width_index:end/2+Width_index)=Gradient_min+((y(end/2-Width_index:end/2+Width_index)+abs(y(end/2-Width_index)))/(2*abs(y(end/2-Width_index))))*(1-Gradient_min); 

    %Limited gradient: option 2 
    %Gradient(end/2-Width_index:end/2+Width_index)=((y(end/2-Width_index:end/2+Width_index)+abs(y(end/2-Width_index)))/(2*abs(y(end/2-Width_index)))); 
    %for p=1:length(Gradient)
    %    if Gradient(p)>0 && Gradient(p)<Gradient_min
    %        Gradient(p)=Gradient_min;
    %    end
    %end

    
    D=asin(Gradient)/pi; %0.25* = reduce to 25 or 50 percent for HCP requirement
    PP_0=fliplr((2*D-1))'*ones(length(Z),1)';
    PP=PP_0;
    for m=1:100
        PP_m=(2/(m*pi))*exp(m*1i*ones(1,length(Gradient))'*abs(Poling_period)*Z).*flipud((sin(pi*m*D)'*ones(length(Z),1)'));
        PP=PP+PP_m+conj(PP_m);
    end
    PP=sign(PP);
    E0=@(P,n,W0) sqrt(P/(2*n*c*eps0*pi*W0^2)) ;
    kappa_i= 2*1i*w_i^2*d33/(k_i*c^2);
    kappa_s= 2*1i*w_s^2*d33/(k_s*c^2);
    
    
% % %     % Create the pixels for DXF:
% % %     PP_BW = PP;
% % %     new_x = linspace(min(x),max(x),length(x)*4); new_dx = new_x(2)-new_x(1);
% % %     new_z = linspace(min(Z),max(Z),length(Z)*2); new_dz = new_z(2)-new_z(1);
% % % 
% % %     [Z2, X2] = meshgrid(new_z,new_x); [Z1, X1] = meshgrid(Z,x);
% % %     PP_BW  = interp2(Z1, X1,PP_BW, Z2, X2);
% % %     PP_BW(PP_BW>0) = 1; PP_BW(PP_BW<0) = 0; %make binary
% % %     Pixels1 = bwboundaries(PP_BW);
% % % 
% % % %     fill polygon with the coordinates of the shapes: x1,y1,x2,y2,x3,y3...
% % %     Pixels = Pixels1;
% % %     for i=1:length(Pixels1)
% % %        Pixels{i}(:,1)=Pixels1{i}(:,1)*new_dx*1e6; %fix according to interpolation
% % %        Pixels{i}(:,2)=Pixels1{i}(:,2)*new_dz*1e6;  % and make in microns
% % %     end
% % %     save(strcat(folder_name, Design_filenames{q}), 'Pixels')

    sim = 1;
    if sim == 1

        E_i=zeros(size(X));
        E_s=zeros(size(X));
        E_p=zeros(size(X));

        E_i_max=zeros(length(Z),1); E_s_max=E_i_max;
        E_along_z=zeros(length(Z),length(X));Ei_along_z=E_along_z; Es_along_z=E_along_z;
        %%
        for n=1:length(Z)

            disp([num2str(100*n/length(Z)),'% gone!']);
            z_tag=Z(n);
            z_tag2=z_tag-Z(1);

            xi_i=2*(z_tag - FocusZ)./b_i;
            tau_i=1./(1+1i*xi_i);

            xi_p=2*(z_tag - FocusZ)./b_p;
            tau_p=1./(1+1i*xi_p);

            xi_s=2*(z_tag - FocusZ)./b_s;
            tau_s=1./(1+1i*xi_s);

            E_p =(E0(PumpPower,n_p,omega0_p)*tau_p)*exp(-(((X-PumpOffsetX).^2./(omega0_p)^2+(Y).^2./(omega0_p)^2).*tau_p)).*exp(1i*k_p*(z_tag - FocusZ));
            if n==1
             E_i =(E0(0.01,n_i,omega0_i)*tau_i)*exp(-((X+95e-6).^2+(Y).^2)./((omega0_i)^2).*tau_i).*exp(1i*k_i*(z_tag - FocusZ));
%             E_s =+(E0(1,n_s,omega0_s)*tau_s)*exp(-((X+100e-6).^2+(Y).^2)./((omega0_s)^2).*tau_s).*exp(1i*k_s*(z_tag - FocusZ));
            end

            %generate the crystal slab at this Z
            PP_xy=ones(length(X),1)*PP(:,n)';

            %Non-linear equations:
            dEi_dz=kappa_i.*PP_xy.*conj(E_p).*E_s;%*exp(-1i*delta_k*z_tag);
            dEs_dz=kappa_s.*conj(PP_xy).*E_p.*E_i;%*exp(1i*delta_k*z_tag);
            %Add the non-linear part
            E_i=E_i+dEi_dz*dz; % update  
            E_s=E_s+dEs_dz*dz;

            %Propagate
            E_i=propagate3(E_i, x, y, k_i, dz); E_i=E_i.*exp(1i*k_i*dz);
            E_s=propagate3(E_s, x, y, k_s, dz); E_s=E_s.*exp(1i*k_s*dz);
            E_i_max(n)=max(max(abs(E_i)));
            E_s_max(n)=max(max(abs(E_s)));
            E_along_z(n,:)=I(E_i(end/2,:),n_i)/w_i+I(E_s(end/2,:),n_s)/w_s;%abs(E_i(end/2,:)).^2+abs(E_s(end/2,:)).^2;
            Ei_along_z(n,:)=I(E_i(end/2,:),n_i)/w_i;%abs(E_i(end/2,:)).^2+abs(E_s(end/2,:)).^2;
            Es_along_z(n,:)=I(E_s(end/2,:),n_s)/w_s;%abs(E_i(end/2,:)).^2+abs(E_s(end/2,:)).^2;
        end

        %%
        pad=9;

        dk_fft=pi/((pad)*MaxX);
        kx_fft=(-((pad)*MaxX/dx-0.5):1:(pad)*MaxX/dx-0.5)*dk_fft; ky_fft=kx_fft;

        k0i=k_i/n_i;     k0s=k_s/n_s;

%         R=0.05;%eigenstate
        R=0.05;%idler ->signal
%         R=0.075;%idler ->signal
% %         E_i=propagate3(E_i, x, y, k0i, 50e-3); 
% %         E_s=propagate3(E_s, x, y, k0s, 50e-3); 
% %         E_i=E_i.*exp(-1i.*k0i.*((X-PumpOffsetX).^2+(Y).^2)./(50e-3));
% %         E_s=E_s.*exp(-1i.*k0s.*((X-PumpOffsetX).^2+(Y).^2)./(50e-3));
% %         E_i=propagate3(E_i, x, y, k0i, 50e-3);
% %         E_s=propagate3(E_s, x, y, k0s, 50e-3);

        dx_farfield_i=dk_fft*R/k0i;
        dx_farfield_s=dk_fft*R/k0s;

        x_farfield_i=(-((pad)*MaxX/dx-0.5):1:(pad)*MaxX/dx-0.5)*dx_farfield_i;y_farfield_i=(-((pad)*MaxX/dx-0.5):1:(pad)*MaxX/dx-0.5)*dx_farfield_i;
        x_farfield_s=(-((pad)*MaxX/dx-0.5):1:(pad)*MaxX/dx-0.5)*dx_farfield_s;y_farfield_s=(-((pad)*MaxX/dx-0.5):1:(pad)*MaxX/dx-0.5)*dx_farfield_s;

        Ei_Far_field_complex=fftshift(fft2(ifftshift(padarray(E_i,4*size(E_i),0))));
        Ei_Far_field=abs(Ei_Far_field_complex);
        %Ei_farfield_avg=Ei_farfield_avg+Ei_Far_field_norm;


        Es_Far_field_complex=fftshift(fft2(ifftshift(padarray(E_s,4*size(E_i),0))));
        Es_Far_field=abs(Es_Far_field_complex);


        figure; imagesc(y,Z,Ei_along_z);
        figure; imagesc(y,Z,Es_along_z);
        figure; imagesc(y,Z,Es_along_z+Ei_along_z);
    end
end
% figure; imagesc(x_farfield_i*10e-3,y_farfield_i*10e-3,I(Ei_Far_field,n_i)/w_i); title('Idler'); xlim([-0.00045 0.00045]); ylim([-0.00045 0.00045]); grid on;
figure; imagesc(x_farfield_s*10e2,y_farfield_s*10e2,I(Es_Far_field,n_s)/w_s);   xlim([-0.00045*10e2 0.00045*10e2]); ylim([-0.00045*10e2 0.00045*10e2]);grid on;
xlabel('X[mm]')
ylabel('Y[mm]')
% title ('Total Intensity Pattern - High Pump Power');
centerofmass=[];
summation=sum(I(Es_Far_field,n_s)/w_s)/size(I(Es_Far_field,n_s)/w_s,1);
X=1:length(summation);
figure,hold on, plot(x_farfield_s,summation/max(summation),'LineWidth',2,'color',[0,0,0.5]);
centerofmass(end+1)=sum(X.*summation)/sum(summation);
%  legend('|\omega_->',,'|\omega_+>');
grid on;
xlim([-0.0004 0.0004]);
xlabel('X[mm]');
ylabel('I[a.u.]');
end
% figure; imagesc(I(E_p, n_p));
