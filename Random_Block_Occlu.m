function [J] = Random_Block_Occlu(I,block_l)
[im_h,im_w]=size(I);

test_tol=2;

wid_rand = rand(test_tol,1);
heig_rand = rand(test_tol,1);
rand_num=[wid_rand;heig_rand];

height     =   floor(sqrt(im_h*im_w*block_l));
width      =   height;

width_rand = rand_num(1:test_tol,:);
height_rand = rand_num(test_tol+1:end,:);

w_a = 1;
w_b=im_w-width+1;
r_w = w_a + (w_b-w_a).*width_rand;
h_a = 1;
h_b=im_h-height+1;
r_h = h_a + (h_b-h_a).*height_rand;


J = I;
baroon = I;
% baroon = baroon./(mean(baroon(:))/mean(I(:)));
r_h    = round(r_h);
r_w    = round(r_w);
J(r_h:r_h+height-1,r_w:r_w+width-1)= zeros(height,width);
