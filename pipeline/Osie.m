file  = load("attrs.mat");
attrs = file.attrs;
attrNames = file.attrNames;
attrs{1, 1}.objs;
for i = 1:700
    current = attrs{i, 1};
    name = current.img;
    disp(name)
    data = zeros(12, 600, 800);
    map = zeros(600, 800);
    for j = 1:length(current.objs)
        current_obj = current.objs{j, 1};
        features = current_obj.features;
        to_map = find(features ~=0);
        % map = map + current_obj.map;
        for k = to_map
            map = reshape(current_obj.map,[1, size(current_obj.map)]);
            data(k, :, :) = data(k, :, :) + map;
        end
     save(sprintf(name(1:4)), 'data')
    end
end