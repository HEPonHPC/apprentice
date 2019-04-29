for iterno = 1:4
  filename_roots = strcat('f20_2x/sincrun/f20_2x_p2_q3_ts2x_d3_lb-6_ub4pi/plots/Croots_iter',int2str(iterno),'.csv')
  outfile = strcat('f20_2x/sincrun/f20_2x_p2_q3_ts2x_d3_lb-6_ub4pi/plots/Prootsplot_iter',int2str(iterno),'.pdf')

  read_roots = csvread(filename_roots);
  X = read_roots;

  x = X(:, 1);
  y = X(:, 2);
  z = X(:, 3);
  K=3
  clr = lines(K);
  G =[]
  % G = [G; 1]
  % G = [G; 2]
  % display(G)
  % display(C)
  newx1 = [];
  newx2 = [];
  newx3 = [];
  newy1 = [];
  newy2 = [];
  newy3 = [];
  newz1 = [];
  newz2 = [];
  newz3 = [];
  for i = 1:length(x)
    if X(i,1)<8 &&  X(i,3)<2
      G = [G; 2];
      newx1 = [newx1,X(i,1)];
      newy1 = [newy1,X(i,2)];
      newz1 = [newz1,X(i,3)];
    elseif X(i,1)<10 &&  X(i,3)>8
      G = [G; 1];
      newx2 = [newx2,X(i,1)];
      newy2 = [newy2,X(i,2)];
      newz2 = [newz2,X(i,3)];
    else
      G = [G; 3];
      newx3 = [newx3,X(i,1)];
      newy3 = [newy3,X(i,2)];
      newz3 = [newz3,X(i,3)];
    end
  end
  figure, hold on
  % scatter3(X(:,1), X(:,2), X(:,3), 36, clr(G,:), 'Marker','.')
    points = 50
    method = 'linear'
    xv = linspace(min(newx1), max(newx1), points);
    yv = linspace(min(newy1), max(newy1), points);
    [XX,YY] = meshgrid(xv, yv);
    ZZ = griddata(newx1,newy1,newz1,XX,YY,method);
    display('plotting')
    surf(XX, YY, ZZ,'FaceColor','green','EdgeColor', 'red','FaceAlpha',0.5);

    xv = linspace(min(newx2), max(newx2), points);
    yv = linspace(min(newy2), max(newy2), points);
    [XX,YY] = meshgrid(xv, yv);
    ZZ = griddata(newx2,newy2,newz2,XX,YY,method);
    display('plotting')
    surf(XX, YY, ZZ,'FaceColor','green','EdgeColor', 'red','FaceAlpha',0.5);

    zv = linspace(min(newz3), max(newz3), points);
    yv = linspace(min(newy3), max(newy3), points);
    [YY,ZZ] = meshgrid(yv, zv);
    XX = griddata(newy3,newz3,newx3,YY,ZZ,method);
    display('plotting')
    surf(XX, YY, ZZ,'FaceColor','green','EdgeColor', 'red','FaceAlpha',0.5);
    % sc = scatter3(1.36538391e+00,1.00000000e-06,1.00000000e-06,400,'black','c','filled')
    % tri = delaunay(newy3,newz3);
    % display('plotting')
    % h = trisurf(tri, newx3, newy3, newz3,'FaceColor','green','EdgeColor', 'none','FaceAlpha',0.5);

  % scatter3(C(:,1), C(:,2), C(:,3), 100, clr, 'Marker','o', 'LineWidth',3)
  view(3), axis vis3d, box on, rotate3d on
%
%
% xv = linspace(min(x), max(x), 100);
% yv = linspace(min(y), max(y), 100);
% [X,Y] = meshgrid(xv, yv);
% Z = griddata(x,y,z,X,Y,'v4');
% surf(X, Y, Z,'FaceColor','green','EdgeColor', 'red','FaceAlpha',0.5);
% grid on
% set(gca, 'ZLim',[0 100])
% shading interp
% scatter3(x,y,z,'Marker','.');
% scatter3(x,y,z,'filled');

%%
% The problem is that the data is made up of individual (x,y,z)
% measurements. It isn't laid out on a rectilinear grid, which is what the
% SURF command expects. A simple plot command isn't very useful.

% plot3(x,y,z,'.-')
% hold on
%% Little triangles
% The solution is to use Delaunay triangulation. Let's look at some
% info about the "tri" variable.
% tri = delaunay(x,y,z);
% % plot(x,y,'.')
% %%
% % How many triangles are there?
% [r,c] = size(tri);
% disp(r)
% %% Plot it with TRISURF
% h = trisurf(tri, x, y, z);
% axis vis3d
%% Clean it up
% axis off
% l = light('Position',[-50 -15 29])
% set(gca,'CameraPosition',[208 -50 7687])
% lighting phong
% shading interp
% colorbar EastOutside


  xlabel('x_{1}','Interpreter','tex','FontSize',17)
  ylabel('x_{2}','Interpreter','tex','FontSize',17)
  zlabel('x_{3}','Interpreter','tex','FontSize',17)
  title(strcat('Iteration No. = ',int2str(iterno)))
  hold off
  print(outfile,'-dpdf','-fillpage');
  clf;
end
% [X11,X22] = meshgrid(X1,X2);
% for iterno = 1:3
%   Y_pq = read_Y_pq(:, iterno);
%   Y_pq = reshape(Y_pq,numel(X1),numel(X2))
%   s = surf(X11,X22,Y_pq,'FaceColor','green','EdgeColor', 'red','FaceAlpha',0.5);
%   hold on
%   xlabel('x_{1}','Interpreter','tex','FontSize',17)
%   ylabel('x_{2}','Interpreter','tex','FontSize',17)
%   zlabel('$\frac{p(x_1,x_2)}{q(x_1,x_2)}$','Interpreter','latex','FontSize',17)
%   a = get(gca,'XTickLabel');
%   b = get(gca,'YTickLabel');
%   c = get(gca,'ZTickLabel');
%   set(gca,'XTickLabel',a,'fontsize',14)
%   set(gca,'YTickLabel',b,'fontsize',14)
%   set(gca,'ZTickLabel',c,'fontsize',14)
%   if(iterno ~= 3)
%     sc = scatter3(polex(1, iterno),polex(2, iterno),polex(3, iterno),400,'black','c','filled')
%     uistack(s,'bottom');
%     uistack(sc,'top');
%   end
%   hold off
%   file = strcat('imap_pq_',int2str(iterno),'.pdf')
%   % print('imap.pdf','-dpdf','-fillpage');
%   print(file,'-dpdf','-fillpage');
% end
%
% Z = zeros(size(X1, 1));
% for iterno = 1:3
%   Z = zeros(size(X1, 1));
%   Y_q = read_Y_q(:, iterno);
%   Y_q = reshape(Y_q,numel(X1),numel(X2))
%   s = surf(X11,X22,Y_q,'FaceColor','green','EdgeColor', 'red','FaceAlpha',0.5);
%   hold on
%   xlabel('x_{1}','Interpreter','tex','FontSize',17)
%   ylabel('x_{2}','Interpreter','tex','FontSize',17)
%   zlabel('$q(x_1,x_2)$','Interpreter','latex','FontSize',17)
%   if(iterno ~= 3)
%     sc = scatter3(polex(1, iterno),polex(2, iterno),polex(3, iterno),400,'black','c','filled')
%     uistack(sc,'top');
%     uistack(s,'bottom');
%   end
%   surf(X1, X2, Z,'FaceColor', [1.0 0.5 0.0], 'EdgeColor', 'none','FaceAlpha',0.3)
%   view(3); camlight; axis vis3d
%   zdiff = Z - Y_q;
%   C = contours(X1, X2, zdiff, [0 0]);
%   xL = C(1, 2:end);
%   yL = C(2, 2:end);
%   zL = interp2(X1, X2, Y_q, xL, yL);
%   line(xL, yL, zL, 'Color', 'blue', 'LineWidth', 5);
%   a = get(gca,'XTickLabel');
%   b = get(gca,'YTickLabel');
%   c = get(gca,'ZTickLabel');
%   set(gca,'XTickLabel',a,'fontsize',14)
%   set(gca,'YTickLabel',b,'fontsize',14)
%   set(gca,'ZTickLabel',c,'fontsize',14)
%   hold off
%   file = strcat('imap_q_',int2str(iterno),'.pdf')
%   % print('imap.pdf','-dpdf','-fillpage');
%   print(file,'-dpdf','-fillpage');
% end

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
