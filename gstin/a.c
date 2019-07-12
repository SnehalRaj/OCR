#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(){
	char str[100];
	scanf("%s",str);
	int len=strlen(str);
	int sum=0,x;
	for(int i=0;i<len-1;i++){
		if(str[i]>='0'&&str[i]<='9'){
				x=((str[i]-'0'));
			}
		else{
			x=((str[i]-'A'+10));
		}
		if(i%2==0){
			sum+=x%36+x/36;
		}
		else{
			x=x*2;
			sum+=(x%36+x/36);
		}
	}
	printf("%d",sum);
	sum=sum%36;
	printf("\n%d\n",sum);
	printf("%d",36-sum);
}
