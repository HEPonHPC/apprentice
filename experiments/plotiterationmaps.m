filename_X = 'f8_noisepct10-1_2x/plots/Cimap_X_f8_noisepct10-1_p4_q3_ts2x.csv';
filename_Y_pq = 'f8_noisepct10-1_2x/plots/Cimap_Y_pq_f8_noisepct10-1_p4_q3_ts2x.csv';
filename_Y_q = 'f8_noisepct10-1_2x/plots/Cimap_Y_q_f8_noisepct10-1_p4_q3_ts2x.csv';
filename_polex = 'f8_noisepct10-1_2x/plots/Cimap_pole_x_f8_noisepct10-1_p4_q3_ts2x.csv';

read_X = csvread(filename_X);
read_Y_pq = csvread(filename_Y_pq);
read_Y_q = csvread(filename_Y_q);
polex = csvread(filename_polex);

X1 = read_X(:, 1);
X2 = read_X(:, 2);
[X11,X22] = meshgrid(X1,X2);
for iterno = 1:3
  Y_pq = read_Y_pq(:, iterno);
  Y_pq = reshape(Y_pq,numel(X1),numel(X2))
  s = surf(X11,X22,Y_pq,'FaceColor','green','EdgeColor', 'red','FaceAlpha',0.5);
  hold on
  xlabel('x_{1}','Interpreter','tex','FontSize',17)
  ylabel('x_{2}','Interpreter','tex','FontSize',17)
  zlabel('$\frac{p(x_1,x_2)}{q(x_1,x_2)}$','Interpreter','latex','FontSize',17)
  a = get(gca,'XTickLabel');
  b = get(gca,'YTickLabel');
  c = get(gca,'ZTickLabel');
  set(gca,'XTickLabel',a,'fontsize',14)
  set(gca,'YTickLabel',b,'fontsize',14)
  set(gca,'ZTickLabel',c,'fontsize',14)
  if(iterno ~= 3)
    sc = scatter3(polex(1, iterno),polex(2, iterno),polex(3, iterno),400,'black','c','filled')
    uistack(s,'bottom');
    uistack(sc,'top');
  end
  hold off
  file = strcat('imap_pq_',int2str(iterno),'.pdf')
  % print('imap.pdf','-dpdf','-fillpage');
  print(file,'-dpdf','-fillpage');
end

Z = zeros(size(X1, 1));
for iterno = 1:3
  Z = zeros(size(X1, 1));
  Y_q = read_Y_q(:, iterno);
  Y_q = reshape(Y_q,numel(X1),numel(X2))
  s = surf(X11,X22,Y_q,'FaceColor','green','EdgeColor', 'red','FaceAlpha',0.5);
  hold on
  xlabel('x_{1}','Interpreter','tex','FontSize',17)
  ylabel('x_{2}','Interpreter','tex','FontSize',17)
  zlabel('$q(x_1,x_2)$','Interpreter','latex','FontSize',17)
  if(iterno ~= 3)
    sc = scatter3(polex(1, iterno),polex(2, iterno),polex(3, iterno),400,'black','c','filled')
    uistack(sc,'top');
    uistack(s,'bottom');
  end
  surf(X1, X2, Z,'FaceColor', [1.0 0.5 0.0], 'EdgeColor', 'none','FaceAlpha',0.3)
  view(3); camlight; axis vis3d
  zdiff = Z - Y_q;
  C = contours(X1, X2, zdiff, [0 0]);
  xL = C(1, 2:end);
  yL = C(2, 2:end);
  zL = interp2(X1, X2, Y_q, xL, yL);
  line(xL, yL, zL, 'Color', 'blue', 'LineWidth', 5);
  a = get(gca,'XTickLabel');
  b = get(gca,'YTickLabel');
  c = get(gca,'ZTickLabel');
  set(gca,'XTickLabel',a,'fontsize',14)
  set(gca,'YTickLabel',b,'fontsize',14)
  set(gca,'ZTickLabel',c,'fontsize',14)
  hold off
  file = strcat('imap_q_',int2str(iterno),'.pdf')
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
