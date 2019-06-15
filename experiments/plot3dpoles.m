filename_X = 'results/plots/Cpoledata.csv';

read_X = csvread(filename_X);
clf();
X1 = read_X(:, 1);
X2 = read_X(:, 2);
X3 = read_X(:, 3);
% s = scatter3(X1,X2,X3,'filled');
s = scatter3(X1,X2,X3,'LineWidth',10)
% view(-30,10)
hold on
xlabel('x_{1}','Interpreter','tex','FontSize',17)
ylabel('x_{2}','Interpreter','tex','FontSize',17)
zlabel('x_{3}','Interpreter','tex','FontSize',17)

xlim([-1 1])
ylim([-1 1])
zlim([-1 1])

a = get(gca,'XTickLabel');
b = get(gca,'YTickLabel');
c = get(gca,'ZTickLabel');
set(gca,'XTickLabel',a,'fontsize',14)
set(gca,'YTickLabel',b,'fontsize',14)
set(gca,'ZTickLabel',c,'fontsize',14)
ax = gca
ax.GridAlpha = 0.9;
pbaspect([1 1 1])

hold off
file = strcat('results/plots/Ppoleplot.pdf')
print(file,'-dpdf','-fillpage');

% quit;
