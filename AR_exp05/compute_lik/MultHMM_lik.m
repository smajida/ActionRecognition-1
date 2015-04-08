function MultHMM_lik(s,K,T)

s
K
T

load(sprintf('../split_posSeq/posSeq_train_K%dT%d.mat', K,T), 'posSeq_train')
load(sprintf('../split_posSeq/posSeq_test_K%dT%d.mat', K,T), 'posSeq_test')
 
tr = cell(9);
em = cell(9);

for i = 1:9
	disp(sprintf('training %d/%d',i,9))
	[tr{i}, em{i}] = hmmtrain(posSeq_train{i}+1, rand(s, s), rand(s, K));
	tr{i} = tr{i} + 1e-8;
	em{i} = em{i} + 1e-8;
end

lik_train = cell(9,1);
for i=1:9
	ntrain(i) = size(posSeq_train{i},1);
	lik_train{i}=zeros(ntrain(i),9);
end
lik_test = zeros(size(posSeq_test,1), 9);

for i = 1:9
	disp(sprintf('decoding %d/%d',i,9))
	for ii = 1:9
		posSeq_ii = posSeq_train{ii};
		for t = 1:ntrain(ii)
			[post, lik_train{ii}(t, i)] = hmmdecode(posSeq_ii(t,:)+1, tr{i}, em{i});
		end
	end
	for t = 1:size(posSeq_test,1)
		[post, lik_test(t, i)] = hmmdecode(posSeq_test(t,:)+1, tr{i}, em{i});
	end
end

save(sprintf('lik_train_s%dK%dT%d.mat', s,K,T), 'lik_train')
save(sprintf('lik_test_s%dK%dT%d.mat', s,K,T), 'lik_test')