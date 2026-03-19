num_epochs = 3 
for epoch in range(num_epochs):     
    for X, y in data_iter:         
        l = loss(net(X) ,y)         
        trainer.zero_grad()         
        l.backward()         
        trainer.step()     
    l = loss(net(features), labels)     
    print(f'epoch {epoch + 1}, loss {l:f}')
