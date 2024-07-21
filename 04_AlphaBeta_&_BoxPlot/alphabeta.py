def minimax(depth,nodInd,maxPlay,values,alpha,beta,path):
    if depth==3:
        return values[nodInd],path+[nodInd]
        
    if maxPlay:
        best=float('-inf')
        best_path=[]
        
        for i in range(2):
            val,new_path=minimax(depth+1,nodInd*2+i,False,values,alpha,beta,path+[nodInd])
            if val>best:
                best=val
                best_path=new_path
                
            alpha=max(alpha,best)
            if alpha>=beta: 
                print(f'Pruned {path+[nodInd]}')
                break
                
        return best,best_path
        
    else:
        best=float('inf')
        best_path=[]
        
        for i in range(2):
            val,new_path=minimax(depth+1,nodInd*2+i,True,values,alpha,beta,path+[nodInd])
            if val<best:
                best=val
                best_path=new_path
                
            beta=min(best,beta)
            if alpha>=beta: 
                print(f'Pruned {path+[nodInd]}')
                break
                
        return best,best_path

# values = [3, 5, 2, 9, 12, 5, 23, 23]
values=[2,3,5,9,0,1,7,5]
res=minimax(0,0,True,values,float('-inf'),float('inf'),[])
print(f'Path: {res[1]} Cost: {res[0]}')
