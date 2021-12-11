function [ skew_x ] = SKEW3( x )
%skew matrix of x
skew_x = [0,-x(3),x(2); x(3),0,-x(1);-x(2),x(1),0];
end

