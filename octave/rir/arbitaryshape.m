% generate the condinates of microphones of arbitrary shaped microphone arrays 
rooms = [ 0 0.1 0 0.1 0 0.1 ];
mic_num = 6; # the number of microphones 
room_size = randi([1,size(rooms,1)]);
mic_X = unifrnd(rooms(room_size,1), rooms(room_size,2));
mic_Y = unifrnd(rooms(room_size,3), rooms(room_size,4));
mic_Z = unifrnd(rooms(room_size,5), rooms(room_size,6));
mic0 = [ mic_X mic_Y mic_Z ];
scales = [ 0.04  0.2 ];
rondom_dis = unifrnd(scales(1), scales(2));
mics = zeros(mic_num, 3);
mics(1,:) = mic0; 
n = 1; 
while n < mic_num
    mic_X = unifrnd(rooms(room_size,1), rooms(room_size,2));
    mic_Y = unifrnd(rooms(room_size,3), rooms(room_size,4));
    mic_Z = unifrnd(rooms(room_size,5), rooms(room_size,6));
    mic_new = [ mic_X mic_Y mic_Z ];
    if norm(mic_new-mics(n,:)) >= rondom_dis
        n = n + 1;
        mics(n,:)= mic_new; 
    end
end 
