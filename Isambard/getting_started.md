# Beginning on Isambard-AI

Once you receive an email from AIRR (Isambard-AI), asking you to join, you can use MyAccessID with your Turing email to make an account.
The details on how to get started are here:
<https://docs.isambard.ac.uk/user-documentation/getting_started/>
You will need to install Clifton, but you can use the same SSH key you would have used for Baskerville.
The project name will be a few random letters and numbers, e.g. "a1b2" would be reached by `ssh a1b2.aip2.isambard`

When you first go on, can you add `umask 0002` to your `~/.bashrc` file?
This will allow others in your project to edit files you create, which is useful for collaboration.

Your home folder can be reached by `~` or `$HOME`, and the project folder is reached by `$PROJECTDIR`.
We strongly recommend you make a folder for each sub-project or personal things (with your name) in the `$PROJECTDIR` directory, so it does not become cluttered.

The `$PROJECTDIR` uses the symbolically linked location, which can cause all sorts of subtle difficulties, so in general I'd recommend using the full path (e.g. `realpath $PROJECTDIR` or `cd -P $PROJECTDIR`).

The `public` folder in the project directory is readable by everyone on Isambard. Your home folder is readable by everyone in your group.

As far as I can tell, you cannot SSH into compute nodes like on Baskerville.
You also don't need to specify the "project" and the "qos" as you did on Baskerville, as the user is linked to the project.
It is possible to use `sattach`, but that is not easy to do something like profiling.
It is possible to use `srun --jobid=<ID> --overlap --pty bash`

It is sometimes necessary to create an SSH key on Isambard-AI (e.g., to access Git repos). 
In this case, I'd recommend generating it on Isambard-AI (so it can be revoked without affecting access from elsewhere) and giving it a strong password.

If you use `uv`, it would be an idea to add a UV install for everyone in the main project folder.
There is a guide for [installing UV on HPC in the tips and tricks section](../tips-and-tricks/uv_install.md).
It can be activated by running `source $PROJECTDIR/uv_install/setup_uv.sh`, and adding that line to your `~/.bashrc` will make uv work when you first log on.
If you do this, you need to remove any `echo` statements so `scp` continues to work.