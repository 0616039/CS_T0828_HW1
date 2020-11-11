# HW1_report
Code for car brand classification.

To read the detailed solution, please, refer to [the kaggle post](https://www.kaggle.com/c/cs-t0828-2020-hw1/overview)
- [Installaton](#First)
- [Create Class File and Download Official Image](#second)
- [Define The Class](#third)
- [Buiding Model](#fourth)
- [Train The Model](#fifth)
- [Predict Testing Data](#sixth)
- [Make Submission](#seventh)

<h2 id="First">Installation</h2>

> using google colab with google drive
<pre><code>auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
download = drive.CreateFile({'id': '1sRUp_jnLMUTBNDiWZYy_qKjjiorDcOt9'})
download.GetContentFile('data.zip')
!unzip data.zip</code></pre>
<h2 id="second">Create Class File and Download Official Image</h2>

> read traing_label.csv and move the image to the correct class file
<pre><code>target_path = './data_test/'
original_path = 'C:/Users/88697/hw1/data/raw/cs-t0828-2020-hw1/training_data/training_data/'
with open('training_labels.csv',"rt", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    rows= [row for row in reader]
    for row in rows:
    	if row[0]!='id':
    		if os.path.exists(target_path+row[1]) :
    			full_path = original_path+row[0]+'.jpg'
    			shutil.move(full_path,target_path + row[1] +'/')
    		else :
    			os.makedirs(target_path+row[1])
    			full_path = original_path+row[0]+'.jpg'
    			shutil.move(full_path,target_path + row[1] +'/')</code></pre>
<h2 id="third">Define The Class</h2>

> Use the folder name to define the class and the class order
<pre><code>def find_classes(dir):
    classes = os.listdir(dir)
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx
classes, c_to_idx = find_classes(data_dir+"/train")
</code></pre>
<h2 id="Fourth">Buiding Model</h2>

> I use the model resnet34 in pytorch
<pre><code>model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 196)</code></pre>
<h2 id="fifth">Train The Model</h2>

> I use 10 epochs and print the tain result every fourty steps
<pre><code>epochs = 10
steps = 0
print_every = 40

model.to('cuda')
model.train()
for e in range(epochs):
    running_loss = 0
    for ii, (inputs, labels) in enumerate(trainloader):
        steps += 1 
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        optimizer.zero_grad()
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step() 
        running_loss += loss.item()
        if steps % print_every == 0:
            model.eval() 
            with torch.no_grad():
                valid_loss, accuracy = validation(model, validloader, criterion)
            model.train()
            lrscheduler.step(accuracy * 100)</code></pre>
<h2 id="sixth">Predict Testing Data</h2>

> predict the test data
<pre><code>a = []
te = []
for i in range(0,16182):
    print(i)
    if os.path.exists('test/My Drive/testing_data/'+str(i).zfill(6)+'.jpg'):
      te.append(i)
      img = image.load_img('test/My Drive/testing_data/'+str(i).zfill(6)+'.jpg', grayscale=False)
      img_preprocessed = preprocess(img)
      batch_img_cat_tensor = torch.unsqueeze(img_preprocessed, 0)
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      batch_img_cat_tensor = batch_img_cat_tensor.to(device)
      model.eval()
      out = model(batch_img_cat_tensor)
      _, index = torch.max(out, 1)
      percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
      a.append(classes[index[0]])</code></pre>
<h2 id="seventh">Make Submission</h2>

> write the prediction into the csv
<pre><code>sample = pd.read_csv('answer.csv')
sample['id'] = te
sample['label'] = a
sample.to_csv('sample_cnn.csv', header=True, index=False)
</code></pre>
