# Contributing to Talos

Thank you very much for taking the effort to contribute Talos. Below you will find some simple and mostly obvious guidelines on how to do it in the most valuable way.

 1. [Ways to Contribute](#ways-to-contribute)

    1.1. [Code](#code)

    1.2. [Ideas](#ideas)

    1.3. [Testing](#testing)

    1.4. [Something Else](#something)

    1.5. [Documentation](#documentation)

  1.6. [Examples](#examples)

 2. [Important Precautions for Code Contributions](#precautions)

    2.1. [Planning](#code)

    2.2. [Testing](#ideas)

    2.3. [Documentation](#docs_for_review)

    2.4. [Branch Management](#branch_management)

 3. [Pull Requests](#pull_requests)

 4. [Specific Guidelines for Github](#github)

## 1. Ways to contribute <a name="ways-to-contribute"></a>

There are several ways programmers, data scientists and others can contribute to Autonomio.

#### 1.1. Contributing Code <a name="code"></a>

##### 1.1.0. Note on Philosophy and Style

**AUTONOMIO DEV PHILOSOPHY**

  - Doing is more interesting than achieving
  - Having fun is more important than being productive
  - Code coverage can, and needs to be 100%
  - User docs are more important than new features
  - Testing is more important than building  
  - Creating great stuff takes long time

**CODING STYLE GUIDELINES**

We follow pep8. Because [reading docs](http://legacy.python.org/dev/peps/pep-0008/) and particulary [style guides](http://legacy.python.org/dev/peps/pep-0008/) more or less suck, and one way is to use Atom and the amazing Linter plugin.

**MORE STYLE GUIDELINES**

We also make the best effort in moving towards following pep20:

  - Beautiful is better than ugly
  - Explicit is better than implicit
  - Simple is better than complex
  - Complex is better than complicated
  - Flat is better than nested
  - Sparse is better than dense
  - Readability counts
  - Special cases aren't special enough to break the rules
  - Although practicality beats purity
  - Errors should never pass silently
  - Unless explicitly silenced
  - In the face of ambiguity, refuse the temptation to guess
  - There should be one-- and preferably only one --obvious way to do it
  - Although that way may not be obvious at first unless you're Dutch
  - Now is better than never
  - Although never is often better than right now
  - If the implementation is hard to explain, it's a bad idea
  - If the implementation is easy to explain, it may be a good idea
  - Namespaces are one honking great idea -- let's do more of those

##### 1.1.1. Contribute to Open Issues

It will be great if you can contribute towards open issues. To do this, the best way is to:

1) check out the [open issues](https://github.com/autonomio/talos/issues)
2) join the conversation and share your willingness to contribute
3) somebody will help you get started / provide more details if needed
4) fork [the current dev](https://github.com/autonomio/talos/issues#fork-destination-box) branch
5) make your changes to your own fork/repo
6) test, test, test
7) if it's a new feature, make changes to test_script.py accordingly
8) make sure that Travis build passes
9) come back and make a pull request

What we really try to avoid, is being this guy...

<img src="https://s-media-cache-ak0.pinimg.com/originals/83/f7/8e/83f78e62feb95acc85d000aaf6350d23.jpg" alt="Drawing" width="300px"/>

#### 1.1.2. Contribute to a New Idea

Same as above, but start by [creating a new issue](https://github.com/autonomio/core-module/issues/new) to open a discussion on the idea you have for contribution.

### 1.2. Contributing Ideas  <a name="ideas"></a>

In case you don't want to contribute code, but have a feature request or some other idea, that is a great contribution as well and will be much appreciated. You can do it by [creating a new issue](https://github.com/autonomio/core-module/issues/new).

<img src="https://mrwweb.com/wp-content/uploads/2012/05/dilbertMay72012-600x186.gif">

### 1.3. Contributing Testing  <a name="testing"></a>

Another great way to contribute is testing, which really just means using Talos and [reporting issues](https://github.com/autonomio/talos/issues/new) as they might arise.

**Testing comes in two forms:**

#### 1.3.1 actual testing

Just use Autonomio for any open challenge you are working on. Or pick one from [Kaggle](https://www.kaggle.com/competitions).

1) Work with Autonomion in data science challenges
2) Try a lot of different things
3) [Report issues](https://github.com/autonomio/talos/issues/new) as you may find them

#### 1.3.2 improving code coverage

We're using [Coveralls](https://coveralls.io) for code coverage testing, and even the smallest contributions to this end help a great deal.

1) Follow the instructions in section 1.1 and 1.3.1
2) Use your own fork to see how the results improve in comparison to [current Master](https://coveralls.io/github/autonomio/core-module)

### 1.4. Contributing Something Else  <a name="something"></a>

Best way to get started might be [starting a discussion](https://github.com/autonomio/talos/issues/new) as they might arise.

### 1.5. Contributing to Manual / Documentation  <a name="documentation"></a>

At the moment there is no manual / documentation, so contributions here would be wonderful. Generally it's better to do something very simple and clear. It seems that [RTD](http://readthedocs.io) is a good option as it can read INDEX.rst in /docs and a slightly more complex but much better looking option would be slate.

<img src="https://stevemiles70.files.wordpress.com/2015/05/dilbertontechnicaldoumentation.png" width="600px">

### 1.6. Contributing Examples  <a name="examples"></a>

One of the most useful ways to contribute is when you use Talos for an actual project / challenge, and then write a blog post about your experience with code examples.

## 2. Important Precautions for Code Contributions <a name="precautions"></a>

### 2.1. Planning the Change <a name="planning"></a>

Before even thinking about making any changes to actual code:

1) Define what is happening now (what needs to be changed)
2) Define what is happening differently (once the code is changed)
3) Use text search to find which files / functions are affected
4) Make sure that you understand what each function is doing in relation to the change

### 2.2. Testing the Change <a name="testing"></a>

Don't commit code that is not thoroughly tested:  

1) Run through the code changes and ask yourself if it makes sense
2) Create a clean environment and install from your fork:

    pip install git+http://your-fork-repo-address.git

3) Perform all the commands where your changes are involved and note them down
4) Change the test_script.py in the repo root with the commands from step 3
5) Make sure that code coverage is not becoming lower*
6) Make sure that Travis build is passed

*In terms of code coverage, 100% coverage for your changes would be ideal. If you can't do that, then at least explain the possible caveats in the commit details and also in the comments section of the pull request you are making.

Once you've gone through all these steps, take a short break, come back and ask yourself the question:

"WHAT COULD GO WRONG?"
### 3. Reviewing Pull Requests <a name="review"></a>

If you've been assigned as a reviewer of a given pull request, unless you've been explicitly asked to do so, **DON'T MERGE** just approve the review and share in the comments what you think. If you don't have any comments, just confirm with a comment that you don't have any. While this is kind of obvious, don't start reviewing before you can see all the tests have passed ;)

### 2.3. Documentation <a name="documentation"></a>

The documentation should:

  - be easy to understand
  - develop together with code (when new functions are added docs are updated)
  - use code examples together with the descriptions

An example of a reasonable quality documentation [here](https://mikkokotila.github.io/slate/#introduction).

### 2.4. Branch Management <a name="branch_management"></a>

  - Nothing ever gets pushed directly to master
  - Merges to master should always be reviewed
  - Features are updated to personal branch and from there to dev
  - Once master is stable, it gets merged with production which updates pypi
  - New release is made from each production merge
  - Personal dev branches may be opened by repo members
  - Non-members should have a private fork


#### 3. Pull Requests <a name="pull_requests"></a>

1. Contributor (user) forks `autonomo/talos` to `origin/talos`.
2. user clones `origin/talos` to `local`, sets upstream branch to point to `autonomio/talos` (via `git remote add upstream ...`, where `...` is the address you see when you click on the git clone button on `autonomio/talos`) and then checks out to `master`.
3. Immediately `git checkout -b my_feature_branch`. All work on a new feature is done on this branch or its children.
4. When work is done on `my_feature_branch`, user can check out to `local/master` and ensure `master` is up to date (`git pull upstream master`), and then merge on their local branch: `git merge --no-ff my_feature_branch`, resolve any merge conflicts, then `git push origin master` and open a PR. This PR will be `origin/master > autonomio/master`.
6. Resolve any conflicts with PR if any remain, then work is done.


### 4. General points on using Github  <a name="github"></a>

1) First things first, make sure you understand [this](https://guides.github.com/introduction/flow/index.html) 100%
2) Also make sure that you clearly understand everything that is said [here](https://blog.hartleybrody.com/git-small-teams/)
3) Working on your local machine, only have one folder (the git remote)
4) Load it as module with:

    import sys
    return sys.path.insert(0, '/home/dev/talos')

5) Frequently fetch origin to make sure you get latest changes from other people
6) Don’t work in separate forks, but in branches
7) Keep commits as small as possible
8) Make clear commit messages (explain what you actually are changing)

For Mac users Github desktop is pretty fantastic. For Linux users the GUIs are not so fantastic. Atom looks like a good cross-platform option.
