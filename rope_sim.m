
% Simulation of a forced rope using a finite element method
N = 100;  % Numberof elements
rho = 6000; % density of rope
L = 1.0;    % length of rope
d = 0.005;  % diameter of rope
E = 35E-6;  % Young's modulus
G = 14E-6; % Shear modulus
eta = 1E3;  % Cconstant of normal daming
eta_bar = 4E-2; % Cconstant of tangential damping

Li = L/N;
    


function ddx = dynamics(q, dq, t)


    M = rho * L * pi * (d/2)^2;  %total mass of rope
    mi = M/N;  %  mass of each element;
    Ji = mi/4 * ( Li^2/3 + d^2 / 4 );  % moment of inertia
    
    
    a = ones(3*N, 1);
    a(1:3:end) = mi;
    a(2:3:end) = mi;
    a(3:3:end) = Ji;
    A = diag(a);
    
    x = q(1:3:end);
    y = q(2:3:end);
    theta = q(3:3:end);
    
    xA = x - Li*cos(theta);
    xB = x + Li*cos(theta);
    
    yA = x - Li*sin(theta);
    yB = x + Li*sin(theta);
    
    % differences in global frame
    dxG = xA(2:end) - xB(1:end-1);
    dyG = yA(2:end) - yB(1:end-1);
    
    phi = atan2(dyG, dxG);
    r = sqrt(dxG.^2 + dyG.^2);
    
    dwx = r * cos(theta(1:end-1) - phi);
    dwy = r * sin(theta(1:end-1) - phi);
    dtheta = theta(2:end) - theta(1:end-1);
    
%     Need to calculate the derivatives of the potential 
    
end