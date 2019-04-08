% filename_X = 'f8_noisepct10-1_2x/plots/Cfnsurf_X_f8_noisepct10-1_p4_q3_ts2x.csv';
% filename_Y = 'f8_noisepct10-1_2x/plots/Cfnsurf_Y_f8_noisepct10-1_p4_q3_ts2x.csv';
% zlimit = [-6.3 2.8]

filename_X = 'f8_2x/plots/Cfnsurf_X_f8_p2_q3_ts2x.csv';
filename_Y = 'f8_2x/plots/Cfnsurf_Y_f8_p2_q3_ts2x.csv';
zlimit = [-16 2]

% filename_X = 'f8_2x/plots/Cfnsurf_X_f8_p3_q3_ts2x.csv';
% filename_Y = 'f8_2x/plots/Cfnsurf_Y_f8_p3_q3_ts2x.csv';
% zlimit = [-12 5]

% filename_X = 'f3_2x/plots/Cfnsurf_X_f3_p4_q3_ts2x.csv';
% filename_Y = 'f3_2x/plots/Cfnsurf_Y_f3_p4_q3_ts2x.csv';
% zlimit = [-7 3]

% filename_X = 'f3_2x/plots/Cfnsurf_X_f3_p5_q6_ts2x.csv';
% filename_Y = 'f3_2x/plots/Cfnsurf_Y_f3_p5_q6_ts2x.csv';
% filename_X = 'f8_2x/plots/Cfnsurf_X_f8_p3_q3_ts2x.csv';
% filename_Y = 'f8_2x/plots/Cfnsurf_Y_f8_p3_q3_ts2x.csv';

% filename_X = 'f9_2x/plots/Cfnsurf_X_f9_p3_q7_ts2x.csv';
% filename_Y = 'f9_2x/plots/Cfnsurf_Y_f9_p3_q7_ts2x.csv';
% zlimit = [-7 3.2]

% filename_X = 'f9_2x/plots/Cfnsurf_X_f9_p4_q4_ts2x.csv';
% filename_Y = 'f9_2x/plots/Cfnsurf_Y_f9_p4_q4_ts2x.csv';
% zlimit = [-12 2.5]

% filename_X = 'f9_noisepct10-1_2x/plots/Cfnsurf_X_f9_noisepct10-1_p5_q5_ts2x.csv';
% filename_Y = 'f9_noisepct10-1_2x/plots/Cfnsurf_Y_f9_noisepct10-1_p5_q5_ts2x.csv';
% zlimit = [-8 3.8]



read_X = csvread(filename_X);
read_Y = csvread(filename_Y);
clf();
X1 = read_X(:, 1);
X2 = read_X(:, 2);
[X11,X22] = meshgrid(X1,X2);
Z = zeros(size(X1, 1));
for index = 1:4
% for index = 1:3
  i1 = mod(index-1,3) + 1
  Y = read_Y(:, i1);
  % Y = read_Y(:, index);
  Y = reshape(Y,numel(X1),numel(X2))
  s = surf(X11,X22,Y,'FaceAlpha',0.5);
  hold on
  colormap(hsv)
  xlabel('x_{1}','Interpreter','tex','FontSize',17)
  ylabel('x_{2}','Interpreter','tex','FontSize',17)
  zlabel('log_{10}(\Delta_{r})','Interpreter','tex','FontSize',17)
  caxis(zlimit)
  xlim([-1 1])
  ylim([-1 1])
  zlim(zlimit)

  if (index == 4)
    cb = colorbar()
    set(groot,'defaultTextInterpreter','latex')
    ylabel(cb, 'log_{10}(\Delta_{r})','FontSize',17)
  end
  a = get(gca,'XTickLabel');
  b = get(gca,'YTickLabel');
  c = get(gca,'ZTickLabel');
  set(gca,'XTickLabel',a,'fontsize',14)
  set(gca,'YTickLabel',b,'fontsize',14)
  set(gca,'ZTickLabel',c,'fontsize',14)


  % view(90,0)
  % surf(X1, X2, Z,'FaceColor', [1.0 0.5 0.0], 'EdgeColor', 'none','FaceAlpha',0.3)
  % view(3); camlight; axis vis3d
  % zdiff = Z - Y;
  % C = contours(X1, X2, zdiff, [0 0]);
  % xL = C(1, 2:end);
  % yL = C(2, 2:end);
  % zL = interp2(X1, X2, Y, xL, yL);
  % line(xL, yL, zL, 'Color', 'blue', 'LineWidth', 5);
  hold off
  file = strcat('fndiff_',int2str(index),'.pdf')
  % print('imap.pdf','-dpdf','-fillpage');
  print(file,'-dpdf','-fillpage');
end

% s = surf(X1,X2,Y_pq,'FaceColor','green','EdgeColor', 'red','FaceAlpha',0.7);
% hold on
% % colormap(mymap)
% colormap(jet)
% % colorbar()
% % view(2)
%
%
% % view(90,0)
% xlabel('x1')
% ylabel('x2')
% scatter(polex(1, iterno),polex(2, iterno),400,'black','h','filled')
% zlabel('Z')

% Z = (99.99*100)* ones(size(X1, 1));
% s2 = surface(X1, X2, Z,'FaceColor', [1.0 0.5 0.0], 'EdgeColor', 'none','FaceAlpha',0.3)
% view(3); camlight; axis vis3d
%
% zdiff = Z - Y_pq;
% C = contours(X1, X2, zdiff, [0 0]);
% xL = C(1, 2:end);
% yL = C(2, 2:end);
% zL = interp2(X1, X2, Y_pq, xL, yL);
% line(xL, yL, zL, 'Color', 'r', 'LineWidth', 3);




quit;
