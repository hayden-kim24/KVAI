# Learning Notes

# Date
2024-07-10
## What I learned
Faker package for generating "fake" data. 
https://github.com/joke2k/faker





# Error Logs

# Error Date & Number
Error #2: 2024-07-10
## Error Status
RESOLVED
## Related Error
Error #1 - 2024-07-05
## Error Type
File Compression + Decompression Issue -- File Content Deleted for data/tab_text/criminal_1970_1995.txt
## Error Description
The file content gets deleted after I decompress the file
## Solution
Keep the file in the local directory. Add it to gitignore so that it won't get added to git. 
## --------- RESOLVED ---------

# Error Date & Number
Error #1: 2024-07-05
## Error Status
RESOLVED
## Error Type
Git Issue -- Large Files
## Error Summary
Git push not woring
## Error Description
"file data/tab_text/criminal_1970_1995.txt:169.98 MB" -> file too big
## Error Message
Enumerating objects: 17, done.
Counting objects: 100% (17/17), done.
Delta compression using up to 8 threads
Compressing objects: 100% (13/13), done.
Writing objects: 100% (15/15), 19.51 MiB | 2.68 MiB/s, done.
Total 15 (delta 0), reused 0 (delta 0), pack-reused 0
remote: error: Trace: 0aed480417aa440d9e419dc5c6e233c00efc71190161bc1bb9dfea6fe91c6ba4
remote: error: See https://gh.io/lfs for more information.
remote: error: File data/tab_text/criminal_1970_1995.txt is 169.98 MB; this exceeds GitHub's file size limit of 100.00 MB
remote: error: GH001: Large files detected. You may want to try Git Large File Storage - https://git-lfs.github.com.
To https://github.com/hayden-kim24/KVAI.git
 ! [remote rejected] main -> main (pre-receive hook declined)
error: failed to push some refs to 'https://github.com/hayden-kim24/KVAI.git'
## Attempted Solution: 
brew install git-lfs
## Follow up Error #1
brew install not working
## Follow up Solution #1:
git lfs migrate info
git lfs migrate import --include="*.txt"
* reference #1: https://github.blog/2017-06-27-git-lfs-2-2-0-released/
* reference #2: https://stackoverflow.com/questions/33330771/git-lfs-this-exceeds-githubs-file-size-limit-of-100-00-mb
## Follow up Error #2:
"batch response: This repository is over its data quota. Account responsible for LFS bandwidth should purchase more data packs to restore access."
## Follow Up Solution #2:
gzip "data/tab_text/criminal_1970_1995.txt"
## --------- RESOLVED ---------




