###はじめに

サーバーとか

- aldebaran
	
- Python 3.5.1

- chainer 1.16.0

####環境設定

virtualenvの準備

pyenvまわり（https://github.com/yyuu/pyenvを参考に）
	
	   
	   $ git clone https://github.com/yyuu/pyenv.git ~/.pyenv
	   
	   $ echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
	  
	   $ echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
	   
	   $ echo 'eval "$(pyenv init -)"' >> ~/.bash_profile
	   
	   $ exec $SHELL
	    
	   $ pyenv install 3.5.1
	   
	
	   
virtualenvまわり（https://github.com/yyuu/pyenv-virtualenvを参考に）
	 
	   $ git clone https://github.com/yyuu/pyenv-virtualenv.git ~/.pyenv/plugins/pyenv-virtualenv
	
	   $ echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bash_profile
	   
	   $ exec $SHELL
	   
#####$ pyenv virtualenv 3.5.1 <仮想環境名> 
で仮想環境作ってください

#####$ pyenv activate <仮想環境名> 

で仮想環境にしてからいろいろしてください	   
	   
	   
chainer一式インストール


	1. $ pip install -U pip
	
	2. $ pip install cython
	
	3. $ pip install chainer
	
	4. $ pip install h5py
	
	

