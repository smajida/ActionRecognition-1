for s = [5, 10, 15]
	for K = [50, 100, 150]
		for T = [10, 30, 50]
		command = sprintf('qsh matlab -nodesktop -singleCompThread -nojvm -r ''"MultHMM_lik(%d,%d,%d); quit;"''', s,K,T);
        unix(command);
	 	end
	end
end