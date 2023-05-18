function nz=nz_KTP_kato(lambda,T)

% based on kato, applied optics 2002
% lambda is in microns
% 0.53 microns <lambda< 1.57 microns


% for n=1:length(lambda)
    nz_no_T_dep=sqrt(4.59423+0.06206./(lambda.^2-0.04763)+110.80672./(lambda.^2-86.12171));

    dT=(T-20);

    dnz=(0.9221./lambda.^3-2.9220./lambda.^2+3.6677./lambda-0.1897)*1e-5*dT;

    nz=nz_no_T_dep+dnz;
end